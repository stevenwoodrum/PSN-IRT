"""Training/evaluation for the proposal's scalar-ability 2PL (Approach 2).

Run via run_proposed_irt.py. Legacy models, feature-transfer experiments,
caches, splits, the submission adapter and Approach 1 artifacts are unchanged.
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
from sklearn.metrics import roc_auc_score

from feature_transfer import load_features
from holdout_training import score
from models.proposed_irt import (A1_SUBJECT_FIELDS, ALL_SUBJECT_FIELDS, MODEL_VERSION,
                                 CategoricalEncoder, ProposedIRT, TokenEncoder)
from models.text_chunking import TEXT_ENCODING
from runtime import allocated_bytes, clear_cache, seed_device


@dataclass(frozen=True)
class Recipe:
    name: str
    fields: tuple = A1_SUBJECT_FIELDS
    item_features: bool = False
    text: bool = True
    hidden: int = 64
    dropout: float = 0.
    objective: str = "bce"


# One change at a time from the A1-field 2PL control.
RECIPES = [
    Recipe("2pl_a1_fields"),
    Recipe("2pl_all_subject_fields", ALL_SUBJECT_FIELDS),
    Recipe("2pl_item_features", item_features=True),
    Recipe("2pl_full", ALL_SUBJECT_FIELDS, item_features=True),
]

DEFAULT_SETTINGS = dict(seed=5814, epochs=100, batch_size=512,
                        learning_rate=3e-4, weight_decay=1e-5, patience=10)
EXTRA_FIELDS = tuple(f for f in ALL_SUBJECT_FIELDS if f not in A1_SUBJECT_FIELDS)


def load_proposed_features(study, raw_root, encoder_root, baseline_manifest):
    """Verified A1 cohort plus the remaining predict() subject fields and item_features.

    load_full_data already checks every raw subjects/items parquet against the
    hashes recorded by Approach 1. subject_id and item_id are join keys only.
    """
    frame, vectors, audit = load_features(study, raw_root, encoder_root, baseline_manifest)
    for field in EXTRA_FIELDS:
        frame[field] = ""
    frame["item_features"] = ""
    feature_keys, identity = {}, []
    for benchmark in sorted(frame.benchmark_id.unique()):
        mask = (frame.benchmark_id == benchmark).to_numpy()
        part = frame.loc[mask]
        subjects = pd.read_parquet(Path(raw_root) / benchmark / "subjects.parquet")
        assert subjects.subject_id.is_unique
        subjects = subjects.set_index("subject_id")
        for field in ALL_SUBJECT_FIELDS:
            values = subjects.loc[part.subject_id, field].fillna("").astype(str).to_numpy()
            if field in A1_SUBJECT_FIELDS:
                assert np.array_equal(values, part[field].to_numpy()), (benchmark, field)
            else:
                frame.loc[mask, field] = values
        items = pd.read_parquet(Path(raw_root) / benchmark / "items.parquet", columns=["item_id", "item_features"])
        assert items.item_id.is_unique
        features = items.set_index("item_id").item_features.fillna("").astype(str)
        frame.loc[mask, "item_features"] = features.loc[part.item_id].to_numpy()
        keys = frame.loc[mask].drop_duplicates("item_id").item_features.str.split(";").explode()
        keys = keys[keys != ""].str.split("=").str[0].value_counts()
        feature_keys[benchmark] = {k: int(v) for k, v in keys.items()}
        part = frame.loc[mask]
        identity.append(dict(benchmark=benchmark, subjects=part.subject_id.nunique(),
                             names=part.normalized_name.nunique(),
                             a1_tuples=len(part[list(A1_SUBJECT_FIELDS)].drop_duplicates()),
                             all_field_tuples=len(part[list(ALL_SUBJECT_FIELDS)].drop_duplicates())))
    by_item = frame.groupby("item_id").item_features.nunique()
    assert (by_item == 1).all(), "An item ID has conflicting item_features."
    audit["subject_fields_all"] = list(ALL_SUBJECT_FIELDS)
    audit["item_feature_keys_by_benchmark"] = feature_keys
    audit["identity_by_benchmark"] = identity
    return frame, vectors, audit


def fit_encoders(train, recipe):
    subject = CategoricalEncoder.fit(train, recipe.fields)
    tokens = (TokenEncoder.fit(train.drop_duplicates("item_id").item_features)
              if recipe.item_features else None)
    return subject, tokens


def make_data(frame, vectors, subject_encoder, token_encoder, recipe, device):
    """Index-based tensors: unique subject records and unique items, gathered per batch."""
    fields = list(recipe.fields)
    keys = pd.MultiIndex.from_frame(frame[fields])
    s_codes, s_unique = pd.factorize(keys)
    subjects = subject_encoder.transform(pd.DataFrame(list(s_unique), columns=fields))
    i_codes, i_unique = pd.factorize(frame.item_id)
    item_rows = frame.drop_duplicates("item_id").set_index("item_id").loc[i_unique]
    parts = []
    if recipe.text:
        parts.append(np.stack([vectors[item] for item in i_unique]).astype(np.float32))
    if recipe.item_features:
        parts.append(token_encoder.transform(item_rows.item_features.tolist()))
    if not parts:
        parts.append(np.ones((len(i_unique), 1), dtype=np.float32))
    items = np.hstack(parts)
    as_t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
    return dict(subjects=as_t(subjects), s_idx=as_t(s_codes, torch.long),
                items=as_t(items), i_idx=as_t(i_codes, torch.long),
                labels=as_t(frame.response.to_numpy(dtype=np.float32)))


def batch_logits(model, data, index):
    return model.logits(data["subjects"][data["s_idx"][index]], data["items"][data["i_idx"][index]])


def predict(model, data):
    model.eval()
    n = len(data["labels"])
    with torch.inference_mode():
        out = [torch.sigmoid(batch_logits(model, data, torch.arange(i, min(i+4096, n), device=data["labels"].device))).cpu().numpy()
               for i in range(0, n, 4096)]
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def within_pair_auc(frame, p):
    """Mean AUC over (benchmark, subject) pairs that contain both outcomes."""
    table = pd.DataFrame(dict(b=frame.benchmark_id.to_numpy(), s=frame.subject_id.to_numpy(),
                              y=frame.response.to_numpy(), p=np.asarray(p)))
    values = [roc_auc_score(g.y, g.p) for _, g in table.groupby(["b", "s"]) if g.y.nunique() == 2]
    return (float(np.mean(values)) if values else np.nan), len(values)


def ece(frame, p, bins=10):
    """Count-weighted |mean prediction - success| over ten fixed bins (row level)."""
    p = np.asarray(p, dtype=float)
    y = frame.response.to_numpy(dtype=float)
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p)) if len(p) else np.nan


def metrics(frame, p):
    auc, pairs = within_pair_auc(frame, p)
    return {**score(frame, p), "within_pair_AUC": auc, "auc_pairs": pairs, "row_ECE": ece(frame, p)}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("This loader accepts only proposed_2pl_scalar_v1 checkpoints.")
    recipe = Recipe(**{**checkpoint["recipe"], "fields": tuple(checkpoint["recipe"]["fields"])})
    subject = CategoricalEncoder(checkpoint["subject_categories"])
    tokens = TokenEncoder(checkpoint["item_vocabulary"]) if recipe.item_features else None
    model = ProposedIRT(subject.dimensions, checkpoint["item_dim"], recipe.hidden, recipe.dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), subject, tokens, recipe, checkpoint


def export_parameters(model, subject_encoder, token_encoder, recipe, frame, vectors, directory, device):
    """Ability per full subject record and (b, a) per item; these reconstruct predictions."""
    model.eval()
    subjects = frame[list(ALL_SUBJECT_FIELDS)].drop_duplicates().copy()
    with torch.inference_mode():
        x = torch.as_tensor(subject_encoder.transform(subjects), device=device)
        subjects["ability"] = model.ability(x).cpu().numpy()
    subjects.to_csv(directory / "subject_abilities.csv", index=False)
    items = frame.drop_duplicates("item_id")
    data = make_data(items, vectors, subject_encoder, token_encoder, recipe, device)
    with torch.inference_mode():
        b, a = model.item_parameters(data["items"][data["i_idx"]])
    table = pd.DataFrame(dict(item_id=items.item_id.to_numpy(), b=b.cpu().numpy(), a=a.cpu().numpy()))
    assert (table.a > 0).all()
    table.to_csv(directory / "item_parameters.csv", index=False)


def evaluate_fold(model, subject_encoder, token_encoder, recipe, split, vectors, directory, name, best_epoch, device):
    training = split["train"]
    mean = float(training.response.mean())
    known_names = set(training.normalized_name)
    seen_records = set(training[list(ALL_SUBJECT_FIELDS)].itertuples(index=False, name=None))
    predictions, comparisons, coverage = [], [], []
    for partition, frame in split.items():
        if partition == "purged" or frame.empty:
            continue
        p = predict(model, make_data(frame, vectors, subject_encoder, token_encoder, recipe, device))
        table = frame.drop(columns="content").copy()
        table["partition"] = partition
        table["known_subject"] = frame.normalized_name.isin(known_names)
        table["known_subject_record"] = [t in seen_records for t in frame[list(ALL_SUBJECT_FIELDS)].itertuples(index=False, name=None)]
        table["probability"] = p
        table["constant_0.5"], table["training_global_mean"] = .5, mean
        predictions.append(table)
        for label, values in ((recipe.name, p), ("constant 0.5", np.full(len(frame), .5)),
                              ("training global mean", np.full(len(frame), mean))):
            comparisons.append(dict(recipe=recipe.name, experiment=name, partition=partition,
                                    predictor=label, rows=len(frame), best_epoch=best_epoch,
                                    known_subject_rows=int(table.known_subject.sum()),
                                    unseen_subject_rows=int((~table.known_subject).sum()),
                                    unseen_subject_record_rows=int((~table.known_subject_record).sum()),
                                    **metrics(frame, values)))
        groups = [("all", np.ones(len(table), dtype=bool)),
                  ("known_name", table.known_subject.to_numpy()), ("unseen_name", ~table.known_subject.to_numpy()),
                  ("known_record", table.known_subject_record.to_numpy()), ("unseen_record", ~table.known_subject_record.to_numpy())]
        for benchmark in sorted(frame.benchmark_id.unique()):
            groups.append(("benchmark:" + benchmark, (frame.benchmark_id == benchmark).to_numpy()))
        for group, mask in groups:
            if mask.any():
                sub = frame.iloc[np.flatnonzero(mask)]
                coverage.append(dict(recipe=recipe.name, experiment=name, partition=partition, group=group,
                                     rows=int(mask.sum()), pairs=int(sub.groupby(["benchmark_id", "subject_id"]).ngroups),
                                     actual_success=float(sub.response.mean()), mean_prediction=float(p[mask].mean()),
                                     **metrics(sub, p[mask])))
    pd.concat(predictions).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(comparisons).to_csv(directory / "comparison.csv", index=False)
    pd.DataFrame(coverage).to_csv(directory / "coverage.csv", index=False)
    pd.concat([f.drop(columns="content").assign(partition=role) for role, f in split.items()]).to_csv(directory / "split.csv", index=False)
    rows = pd.concat([f for role, f in split.items() if role != "purged"])
    export_parameters(model, subject_encoder, token_encoder, recipe, rows, vectors, directory, device)
    return pd.DataFrame(comparisons)


def run_fold(split, vectors, directory, settings, recipe, name, device="cpu", score_outer=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    seed_device(seed, device)
    train, validation = split["train"], split["validation"]
    subject_encoder, token_encoder = fit_encoders(train, recipe)
    data = make_data(train, vectors, subject_encoder, token_encoder, recipe, device)
    val = make_data(validation, vectors, subject_encoder, token_encoder, recipe, device)
    item_dim = data["items"].shape[1]
    model = ProposedIRT(subject_encoder.dimensions, item_dim, recipe.hidden, recipe.dropout).to(device)
    devices = sorted({str(p.device) for p in model.parameters()} | {str(t.device) for t in (*data.values(), *val.values())})
    if any(torch.device(d).type != torch.device(device).type for d in devices):
        raise RuntimeError("Parameters/tensors are not on the requested device.")
    allocated = allocated_bytes(device)
    print(f"{recipe.name}/{name}: device={devices}, subject_dim={subject_encoder.dimensions}, item_dim={item_dim}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=.5)
    best, stale, state, best_epoch, history = float("inf"), 0, None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for index in torch.randperm(len(train), device=device).split(settings["batch_size"]):
            optimizer.zero_grad()
            z = batch_logits(model, data, index)
            if recipe.objective != "bce":
                raise ValueError("This variant trains with BCE only.")
            loss = F.binary_cross_entropy_with_logits(z, data["labels"][index])
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
            print(f"{recipe.name}/{name}: epoch {epoch}, train row={tr['row_Brier']:.4f}, val pair={monitored:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    torch.save(dict(model_version=MODEL_VERSION, recipe=asdict(recipe),
                    subject_categories=subject_encoder.categories,
                    item_vocabulary=token_encoder.vocabulary if token_encoder else [],
                    item_dim=item_dim, state_dict=state, text_encoding=TEXT_ENCODING,
                    best_epoch=best_epoch, seed=seed), directory / "best_model.pt")
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    # Outer rows stay unscored for item trials until the runner locks selection.
    visible = split if score_outer else {k: v for k, v in split.items() if k != "heldout"}
    result = evaluate_fold(model, subject_encoder, token_encoder, recipe, visible, vectors, directory, name, best_epoch, device)
    (directory / "settings.json").write_text(json.dumps(dict(
        **settings, recipe=asdict(recipe), model_version=MODEL_VERSION,
        subject_dimensions=subject_encoder.dimensions, item_dim=item_dim,
        item_vocabulary_size=token_encoder.dimensions if token_encoder else 0,
        training_device=str(next(model.parameters()).device), training_tensor_devices=devices,
        accelerator_allocated_bytes_before_training=allocated, best_epoch=best_epoch, epochs_run=len(history),
        checkpoint_selection="internal validation equal subject-ID/benchmark-pair Brier",
        objective="BCE with logits (row-weighted)", outer_scored=score_outer, seconds=time.monotonic()-started,
        metadata_policy="Train-only categories/vocabulary; missing -> empty string; unknown -> zeros; no IDs or outcomes as features.",
    ), indent=2))
    del model, data, val
    clear_cache(device)
    return result
