"""Reproducible item/benchmark holdouts for the chunked Semantic IRT notebook."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from models.Semantic_IRT import Semantic_IRT
from models.text_chunking import TEXT_ENCODING, embed_text, make_text_encoder


def load_samples(source, chunked_path, encoder_source, device="cpu"):
    """Re-embed the exact saved sample once; validate a reusable cache's provenance."""
    source, chunked_path = Path(source), Path(chunked_path)
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    samples = json.loads(raw)
    required = {"benchmark_id", "subject_id", "normalized_name", "item_id", "response_id", "content", "response"}
    if not samples or any(not required <= sample.keys() for sample in samples):
        raise ValueError("Samples must contain the original response, subject and item fields.")
    if any(not isinstance(s["content"], str) or not isinstance(s["normalized_name"], str)
           or s["response"] not in (0, 1) for s in samples):
        raise ValueError("Expected string prompts/subject names and binary responses.")
    if chunked_path.exists():
        cached = json.loads(chunked_path.read_text(encoding="utf-8"))
        if len(cached) != len(samples):
            raise ValueError("Chunked sample count differs from the original.")
        for original, saved in zip(samples, cached):
            if saved.get("text_encoding") != TEXT_ENCODING or any(
                saved.get(key) != original[key] for key in required
            ):
                raise ValueError("Chunked samples differ from the original rows or encoding settings.")
        samples = cached
        print(f"Loaded {len(samples):,} cached chunked responses.")
    else:
        prompts = list(dict.fromkeys(s["content"] for s in samples))
        encoder = make_text_encoder(
            encoder_source, TEXT_ENCODING["text_max_length"], device=device,
            chunk_size=TEXT_ENCODING["text_chunk_size"],
        )
        vectors = embed_text(encoder, prompts, batch_size=32, show_progress=True)
        by_text = dict(zip(prompts, vectors.tolist()))
        for sample in samples:
            sample["content_embedding"] = by_text[sample["content"]]
            sample["text_encoding"] = TEXT_ENCODING
        with chunked_path.open("x", encoding="utf-8") as file:
            json.dump(samples, file, ensure_ascii=False, indent=2)
        print(f"Saved {len(samples):,} responses / {len(prompts):,} prompt vectors to {chunked_path}.")

    item_vectors = {}
    for sample in samples:
        vector = np.asarray(sample["content_embedding"], dtype=np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("Expected finite, normalized 384-dimensional embeddings.")
        if sample["item_id"] in item_vectors and not np.array_equal(item_vectors[sample["item_id"]], vector):
            raise ValueError("One item ID has inconsistent embeddings.")
        item_vectors[sample["item_id"]] = vector
    frame = pd.DataFrame([{key: s[key] for key in required} for s in samples])
    if frame.duplicated(["benchmark_id", "response_id"]).any():
        raise ValueError("Duplicate response identities would distort the scores.")
    frame["row_id"] = np.arange(len(frame))
    frame["item_group"] = item_groups(frame)
    return frame, item_vectors, source_hash


def item_groups(frame):
    """Keep repeated item IDs and identical nonempty prompts together, as in Approach 1."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for item_id, content in frame[["item_id", "content"]].itertuples(index=False, name=None):
        item_key = "item:" + str(item_id)
        text = content.strip()
        if text:
            text_key = "text:" + hashlib.sha256(text.encode()).hexdigest()
            parent[find(item_key)] = find(text_key)
        else:
            find(item_key)
    return frame["item_id"].map(lambda item: find("item:" + str(item)))


def make_split(frame, seed, validation_fraction=0.2, benchmark=None):
    """Benchmark rows never enter training or checkpoint selection; purge shared prompts."""
    heldout = frame.iloc[:0].copy() if benchmark is None else frame[frame.benchmark_id == benchmark].copy()
    pool = frame.copy() if benchmark is None else frame[frame.benchmark_id != benchmark].copy()
    duplicate_mask = pool.item_group.isin(heldout.item_group)
    purged = pool[duplicate_mask].copy()
    pool = pool[~duplicate_mask].copy()
    groups = sorted(pool.item_group.unique())
    if len(groups) < 2:
        raise ValueError("Need at least two independent item/prompt groups to split.")
    training_groups, validation_groups = train_test_split(groups, test_size=validation_fraction, random_state=seed)
    train = pool[pool.item_group.isin(training_groups)].copy()
    validation = pool[pool.item_group.isin(validation_groups)].copy()
    parts = (train, validation, heldout, purged)
    assert sum(map(len, parts)) == len(frame)
    for i, left in enumerate(parts):
        for right in parts[i + 1:]:
            assert set(left.row_id).isdisjoint(right.row_id)
    for left, right in ((train, validation), (train, heldout), (validation, heldout)):
        assert set(left.item_group).isdisjoint(right.item_group)
        assert set(left.item_id).isdisjoint(right.item_id)
    if benchmark is not None:
        assert benchmark not in set(train.benchmark_id) | set(validation.benchmark_id)
        assert set(heldout.benchmark_id) == {benchmark}
    return {"train": train, "validation": validation, "heldout": heldout, "purged": purged}


def score(frame, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(frame) != len(probabilities) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Expected one finite probability in [0, 1] per response.")
    if frame.empty:
        return {"row_Brier": np.nan, "equal_pair_Brier": np.nan, "BCE": np.nan}
    errors = frame[["benchmark_id", "subject_id"]].copy()
    y = frame.response.to_numpy(dtype=float)
    errors["error"] = (probabilities - y) ** 2
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "row_Brier": float(errors.error.mean()),
        "equal_pair_Brier": float(errors.groupby(["benchmark_id", "subject_id"]).error.mean().mean()),
        "BCE": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
    }


def tensors(frame, item_vectors, student_indices, device):
    indices = frame.normalized_name.map(student_indices).fillna(-1).to_numpy(dtype=int)
    known = torch.tensor(indices >= 0, device=device)
    one_hot = torch.zeros((len(frame), len(student_indices)), device=device)
    if known.any():
        one_hot[known] = F.one_hot(torch.tensor(indices[indices >= 0], device=device), len(student_indices)).float()
    vectors = torch.as_tensor(np.stack([item_vectors[item] for item in frame.item_id]), dtype=torch.float32, device=device)
    labels = torch.tensor(frame.response.to_numpy(dtype=np.float32), device=device)
    return one_hot, vectors, labels, known


def predict_tensors(model, data):
    """Use the existing population-mean ability fallback for unseen subject names."""
    subjects, vectors, _, known = data
    model.eval()
    with torch.inference_mode():
        predictions = torch.empty(len(vectors), device=vectors.device)
        if known.any():
            predictions[known] = model(subjects[known], vectors[known]).flatten()
        if (~known).any():
            theta = model.student_net(torch.eye(subjects.shape[1], device=vectors.device)).mean()
            params = model.item_param_head(model.item_embed_proj(vectors[~known]))
            beta = params[:, 0]
            a = torch.ones_like(beta) if model.model_type == "1pl" else F.softplus(params[:, 1]) + 0.1
            c = torch.zeros_like(beta) if model.model_type in ("1pl", "2pl") else torch.sigmoid(params[:, 2]) * 0.3
            d = torch.sigmoid(params[:, 3]) if model.model_type == "4pl" else torch.ones_like(beta)
            predictions[~known] = c + (d - c) * torch.sigmoid(a * (theta - beta))
    return predictions.cpu().numpy()


def run_fold(split, item_vectors, directory, settings, name, device="cpu"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, validation = split["train"], split["validation"]
    students = {name: i for i, name in enumerate(dict.fromkeys(train.normalized_name))}
    model = Semantic_IRT(len(students), item_embed_dim=384, hidden_dim=64, model_type="4pl").to(device)
    train_data = tensors(train, item_vectors, students, device)
    val_data = tensors(validation, item_vectors, students, device)
    assert train_data[3].all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
    best, stale, best_epoch, history, best_state = float("inf"), 0, 0, [], None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        indices = torch.randperm(len(train), device=device)
        for batch in indices.split(settings["batch_size"]):
            optimizer.zero_grad()
            p = model(train_data[0][batch], train_data[1][batch]).flatten()
            loss = F.mse_loss(p, train_data[2][batch])
            loss.backward()
            optimizer.step()
        # Both curves use the same epoch-end weights with model.eval(), as in Approach 1.
        training = score(train, predict_tensors(model, train_data))
        valid = score(validation, predict_tensors(model, val_data))
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"training_{key}": value for key, value in training.items()},
                        **{f"validation_{key}": value for key, value in valid.items()}})
        monitored = valid["equal_pair_Brier"]
        scheduler.step(monitored)
        if monitored < best:
            best, stale, best_epoch, best_state = monitored, 0, epoch, deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= settings["patience"]:
            print(f"{name}: epoch {epoch:3} | train Brier {training['row_Brier']:.4f} | validation pair Brier {monitored:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "student_indices": students, "text_encoding": TEXT_ENCODING,
                "model_type": "4pl", "best_epoch": best_epoch}, directory / "best_model.pt")
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)

    prediction_frames, comparisons = [], []
    training_mean = float(train.response.mean())
    for partition, frame in split.items():
        if frame.empty or partition == "purged":
            continue
        data = tensors(frame, item_vectors, students, device)
        probabilities = predict_tensors(model, data)
        known = data[3].cpu().numpy()
        prediction = frame.drop(columns=["content"]).copy()
        prediction["partition"] = partition
        prediction["known_subject"] = known
        prediction["probability"] = probabilities
        prediction["constant_0.5"] = 0.5
        prediction["training_global_mean"] = training_mean
        prediction_frames.append(prediction)
        for predictor, values in (("Semantic IRT", probabilities), ("constant 0.5", np.full(len(frame), 0.5)),
                                  ("training global mean", np.full(len(frame), training_mean))):
            comparisons.append({"experiment": name, "partition": partition, "predictor": predictor,
                                "rows": len(frame), "items": frame.item_id.nunique(),
                                "known_subject_rows": int(known.sum()), "unseen_subject_rows": int((~known).sum()),
                                "best_epoch": best_epoch, **score(frame, values),
                                "known_only_row_Brier": score(frame.loc[known], np.asarray(values)[known])["row_Brier"]})
    pd.concat(prediction_frames).to_csv(directory / "predictions.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(directory / "comparison.csv", index=False)
    ledger = pd.concat([part.drop(columns="content").assign(partition=label) for label, part in split.items()])
    ledger.to_csv(directory / "split.csv", index=False)
    (directory / "settings.json").write_text(json.dumps({**settings, "text_encoding": TEXT_ENCODING,
        "checkpoint_selection": "internal validation equal subject-benchmark-pair Brier",
        "objective": "response-row MSE/Brier", "subject_fallback": "mean training-subject ability",
        "purged_rows": len(split["purged"]), "epochs_run": len(history)}, indent=2))
    print(f"{name}: selected epoch {best_epoch}; saved {directory}", flush=True)
    return comparison


def learning_curves(run_dir):
    import matplotlib.pyplot as plt

    paths = sorted(Path(run_dir).glob("*/training_history.csv"))
    fig, axes = plt.subplots(len(paths), 1, figsize=(11, 3.1 * len(paths)), squeeze=False)
    for ax, path in zip(axes.flat, paths):
        history = pd.read_csv(path)
        comparison = pd.read_csv(path.parent / "comparison.csv")
        for column, label, color in (("training_row_Brier", "Training (row)", "tab:blue"),
                                      ("validation_row_Brier", "Validation (row)", "tab:orange"),
                                      ("validation_equal_pair_Brier", "Validation (equal pair)", "tab:green")):
            ax.plot(history.epoch, history[column], label=label, color=color)
        ax.axvline(int(comparison.best_epoch.iloc[0]), color="black", linestyle=":", label="Selected epoch")
        ax.set(title=path.parent.name.replace("__", ": "), xlabel="Epoch", ylabel="Brier loss (lower is better)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Semantic IRT — epoch-end training and internal validation", y=1.002)
    fig.tight_layout()
    return fig


def comparison_plot(comparison, partition):
    import matplotlib.pyplot as plt

    data = comparison[comparison.partition == partition]
    wide = data.pivot(index="experiment", columns="predictor", values="equal_pair_Brier")
    order = (["item_holdout"] if "item_holdout" in wide.index else []) + [name for name in wide.index if name != "item_holdout"]
    wide = wide.loc[order]
    fig, ax = plt.subplots(figsize=(12, 5))
    wide[["constant 0.5", "training global mean", "Semantic IRT"]].plot.bar(ax=ax, rot=0)
    names = {"multi_swebench": "Multi-SWE-bench", "real_webagents": "WebAgents",
             "researchcodebench": "ResearchCodeBench", "swe_rebench": "SWE-rebench", "item_holdout": "Item holdout"}
    ax.set_xticklabels([("Hold out:\n" if name.startswith("benchmark__") else "") +
                        names.get(name.removeprefix("benchmark__"), name) for name in wide.index])
    ax.set(xlabel="", ylabel="Equal-pair Brier (lower is better)",
           title="Internal validation at selected checkpoint" if partition == "validation" else
           "Entirely held-out benchmark — scored after checkpoint selection")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.18)
    ax.legend(title=None, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3)
    fig.tight_layout()
    return fig
