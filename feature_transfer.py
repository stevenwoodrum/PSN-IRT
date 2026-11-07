"""Controlled transfer of A1 features into a new Semantic IRT variant.

Run via run_feature_transfer.py. Original models, experiments,
embeddings, submission adapter and data remain unchanged.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from full_data_training import load_full_data, recorded_split, sha
from holdout_training import score
from models.feature_irt import FeatureIRT, MetadataEncoder, MODEL_VERSION, SUBJECT_FIELDS
from models.text_chunking import TEXT_ENCODING
from runtime import allocated_bytes, clear_cache, seed_device


@dataclass(frozen=True)
class Recipe:
    name: str
    fields: tuple = SUBJECT_FIELDS
    subject_mode: str = "metadata"
    text: bool = True
    hidden: int = 64
    dropout: float = 0.
    kernel_l2: float = 0.
    objective: str = "mse"


RECIPES = [
    Recipe("ordered_name", ("normalized_name",), "name_mean"),
    Recipe("full_metadata"),
    Recipe("no_name", ("provider", "reasoning_effort")),
    Recipe("no_provider", ("normalized_name", "reasoning_effort")),
    Recipe("no_reasoning", ("normalized_name", "provider")),
    Recipe("no_text", text=False),
    Recipe("dropout_02", dropout=.2),
    Recipe("kernel_l2_0001", kernel_l2=1e-4),
    Recipe("hidden_32", hidden=32),
    Recipe("bce", objective="bce"),
]

DEFAULT_SETTINGS = dict(seed=5814, epochs=100, batch_size=512,
                        learning_rate=3e-4, weight_decay=1e-5, patience=10)


def load_features(study, raw_root, encoder_root, baseline_run):
    baseline_path = Path(baseline_run)
    if baseline_path.is_dir():
        baseline_path = baseline_path / "manifest.json"
    baseline = json.loads(baseline_path.read_text())
    for file, expected in baseline["data_audit"]["source_sha256"].items():
        if sha(Path(study) / file) != expected:
            raise ValueError(f"Cohort input differs from frozen baseline: {file}")
    frame, vectors, audit = load_full_data(study, raw_root, encoder_root)
    columns = ["row_id", "subject_provider", "subject_reasoning_effort"]
    metadata = pd.read_parquet(Path(study) / "observations.parquet", columns=columns)
    assert np.array_equal(metadata.row_id, frame.row_id)
    for field in ("provider", "reasoning_effort"):
        frame[field] = metadata["subject_" + field].fillna("").astype(str)
    for benchmark in frame.benchmark_id.unique():
        part = frame[frame.benchmark_id == benchmark]
        subjects = pd.read_parquet(Path(raw_root) / benchmark / "subjects.parquet").set_index("subject_id")
        for field in SUBJECT_FIELDS:
            actual = subjects.loc[part.subject_id, field].fillna("").astype(str).to_numpy()
            if not np.array_equal(actual, part[field].to_numpy()):
                raise ValueError(f"Subject metadata does not match raw source: {benchmark}/{field}")
    audit["subject_fields"] = list(SUBJECT_FIELDS)
    audit["metadata_verified_against_raw"] = True
    return frame, vectors, audit


def identity_audit(frame):
    rows = []
    for benchmark, part in frame.groupby("benchmark_id"):
        rows.append(dict(benchmark=benchmark, rows=len(part), subjects=part.subject_id.nunique(),
                         names=part.normalized_name.nunique(),
                         metadata_tuples=len(part[list(SUBJECT_FIELDS)].drop_duplicates()),
                         blank_name_rows=int(part.normalized_name.eq("").sum())))
    return pd.DataFrame(rows)


def make_tensors(frame, item_vectors, encoder, recipe, device):
    subjects = torch.from_numpy(encoder.transform(frame)).to(device)
    # One dummy constant replaces all text for the no-text control; item IDs
    # are used only to retrieve vectors, never as learned input categories.
    items = (np.stack([item_vectors[item] for item in frame.item_id]) if recipe.text
             else np.ones((len(frame), 1), dtype=np.float32))
    items = torch.as_tensor(items, dtype=torch.float32, device=device)
    labels = torch.tensor(frame.response.to_numpy(dtype=np.float32), device=device)
    return subjects, items, labels


def predict(model, data):
    model.eval()
    with torch.inference_mode():
        values = [model(data[0][i:i+4096], data[1][i:i+4096]).cpu().numpy()
                  for i in range(0, len(data[0]), 4096)]
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("This loader accepts only ordered metadata v1 checkpoints, not legacy weights.")
    encoder = MetadataEncoder(checkpoint["categories"])
    recipe = Recipe(**{**checkpoint["recipe"], "fields": tuple(checkpoint["recipe"]["fields"])})
    model = FeatureIRT(encoder.dimensions, checkpoint["item_dim"], recipe.hidden,
                       recipe.dropout, recipe.subject_mode).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), encoder, recipe, checkpoint


def export_parameters(model, encoder, frame, vectors, recipe, directory, device):
    model.eval()
    subjects = frame[list(SUBJECT_FIELDS)].drop_duplicates().copy()
    with torch.inference_mode():
        subjects["ability"] = model.ability(torch.from_numpy(encoder.transform(subjects)).to(device)).cpu().numpy()
    subjects.to_csv(directory / "subject_abilities.csv", index=False)
    items = frame.drop_duplicates("item_id").copy()
    parts = []
    for start in range(0, len(items), 4096):
        part = items.iloc[start:start+4096]
        data = make_tensors(part, vectors, encoder, recipe, device)
        with torch.inference_mode():
            params = model.item_parameters(data[1])
        table = part[["item_id"]].copy()
        for field, values in zip(("beta", "a", "c", "d"), params):
            table[field] = values.cpu().numpy()
        parts.append(table)
    table = pd.concat(parts, ignore_index=True)
    assert (table.a > 0).all() and (table.c >= 0).all()
    assert (table.d >= table.c).all() and (table.d <= 1).all()
    table.to_csv(directory / "item_parameters.csv", index=False)


def evaluate_fold(model, encoder, recipe, split, vectors, directory, name, best_epoch, device):
    training = split["train"]
    mean = float(training.response.mean())
    known_names = set(training.normalized_name)
    seen_tuples = set(training[list(SUBJECT_FIELDS)].itertuples(index=False, name=None))
    predictions, comparisons, coverage = [], [], []
    for partition, frame in split.items():
        if partition == "purged" or frame.empty:
            continue
        data = make_tensors(frame, vectors, encoder, recipe, device)
        p = predict(model, data)
        table = frame.drop(columns="content").copy()
        table["partition"] = partition
        table["known_subject"] = frame.normalized_name.isin(known_names)
        table["known_metadata_tuple"] = [t in seen_tuples for t in frame[list(SUBJECT_FIELDS)].itertuples(index=False, name=None)]
        for field in SUBJECT_FIELDS:
            table["known_" + field] = frame[field].isin(set(training[field]))
        table["probability"] = p
        table["constant_0.5"], table["training_global_mean"] = .5, mean
        predictions.append(table)
        for label, values in ((recipe.name, p), ("constant 0.5", np.full(len(frame), .5)),
                              ("training global mean", np.full(len(frame), mean))):
            comparisons.append(dict(recipe=recipe.name, experiment=name, partition=partition,
                                    predictor=label, rows=len(frame), best_epoch=best_epoch,
                                    known_subject_rows=int(table.known_subject.sum()),
                                    unseen_subject_rows=int((~table.known_subject).sum()), **score(frame, values)))
        for group, mask in (("all", np.ones(len(table), dtype=bool)),
                            ("known_name", table.known_subject.to_numpy()),
                            ("unseen_name", ~table.known_subject.to_numpy())):
            if mask.any():
                coverage.append(dict(recipe=recipe.name, experiment=name, partition=partition,
                                     group=group, rows=int(mask.sum()), actual_success=float(frame.response.to_numpy()[mask].mean()),
                                     mean_prediction=float(p[mask].mean()), **score(frame.iloc[np.flatnonzero(mask)], p[mask])))
    pd.concat(predictions).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(comparisons).to_csv(directory / "comparison.csv", index=False)
    pd.DataFrame(coverage).to_csv(directory / "coverage.csv", index=False)
    pd.concat([p.drop(columns="content").assign(partition=role) for role, p in split.items()]).to_csv(directory / "split.csv", index=False)
    all_rows = pd.concat([p for role, p in split.items() if role != "purged"])
    export_parameters(model, encoder, all_rows, vectors, recipe, directory, device)
    return pd.DataFrame(comparisons)


def run_fold(split, vectors, directory, settings, recipe, name, device="cpu", score_outer=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    seed_device(seed, device)
    train, validation = split["train"], split["validation"]
    encoder = (MetadataEncoder({"normalized_name": list(dict.fromkeys(train.normalized_name))})
               if recipe.subject_mode == "name_mean" else MetadataEncoder.fit(train, recipe.fields))
    item_dim = len(next(iter(vectors.values()))) if recipe.text else 1
    model = FeatureIRT(encoder.dimensions, item_dim, recipe.hidden, recipe.dropout, recipe.subject_mode).to(device)
    data = make_tensors(train, vectors, encoder, recipe, device)
    val = make_tensors(validation, vectors, encoder, recipe, device)
    devices = sorted({str(p.device) for p in model.parameters()} | {str(t.device) for t in (*data, *val)})
    if any(torch.device(d).type != torch.device(device).type for d in devices):
        raise RuntimeError("Parameters/tensors are not on the requested device.")
    allocated = allocated_bytes(device)
    print(f"{recipe.name}/{name}: device={devices}, allocated bytes={allocated}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=.5)
    best, stale, state, best_epoch, history = float("inf"), 0, None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for indices in torch.randperm(len(train), device=device).split(settings["batch_size"]):
            optimizer.zero_grad()
            p = model(data[0][indices], data[1][indices])
            labels = data[2][indices]
            loss = (F.mse_loss(p, labels) if recipe.objective == "mse"
                    else F.binary_cross_entropy(p.clamp(1e-7, 1-1e-7), labels))
            if recipe.kernel_l2:
                loss = loss + recipe.kernel_l2 * model.kernel_penalty()
            loss.backward()
            optimizer.step()
        tr, va = score(train, predict(model, data)), score(validation, predict(model, val))
        monitored = va["equal_pair_Brier"]
        if not np.isfinite(monitored):
            raise RuntimeError("Nonfinite validation metric.")
        history.append(dict(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"],
                            **{"training_" + k: v for k, v in tr.items()},
                            **{"validation_" + k: v for k, v in va.items()}))
        scheduler.step(monitored)
        if monitored < best:
            best, stale, state, best_epoch = monitored, 0, deepcopy(model.state_dict()), epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= settings["patience"]:
            print(f"{recipe.name}/{name}: epoch {epoch}, train={tr['row_Brier']:.4f}, val pair={monitored:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    torch.save(dict(model_version=MODEL_VERSION, recipe=asdict(recipe), categories=encoder.categories,
                    item_dim=item_dim, state_dict=state, text_encoding=TEXT_ENCODING,
                    best_epoch=best_epoch), directory / "best_model.pt")
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    # The runner persists a recipe-selection receipt BEFORE any outer test is
    # scored. All item trials are fitted using train/validation alone first.
    visible = split if score_outer else {k: v for k, v in split.items() if k != "heldout"}
    result = evaluate_fold(model, encoder, recipe, visible, vectors, directory, name, best_epoch, device)
    (directory / "settings.json").write_text(json.dumps(dict(
        **settings, recipe=asdict(recipe), model_version=MODEL_VERSION, categories=encoder.categories,
        training_device=str(next(model.parameters()).device), training_tensor_devices=devices,
        accelerator_allocated_bytes_before_training=allocated, best_epoch=best_epoch, epochs_run=len(history),
        checkpoint_selection="internal validation equal subject-ID/benchmark-pair Brier",
        objective=recipe.objective, outer_scored=score_outer, seconds=time.monotonic()-started,
        metadata_policy="Train-only field categories; missing canonicalized to empty string; unknown field all zeros.",
    ), indent=2))
    del model, data, val
    clear_cache(device)
    return result
