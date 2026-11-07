"""Full-cohort feature ablations on CUDA, Apple GPU, or CPU."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import torch

from feature_transfer import (DEFAULT_SETTINGS, RECIPES, evaluate_fold, identity_audit,
                              load_checkpoint, load_features, run_fold)
from full_data_training import recorded_split, sha
from models.feature_irt import MODEL_VERSION
from runtime import clear_cache, select_device


ROOT = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("PSN_IRT_STUDY", ROOT.parent / "my-approach1/experiments/approach1_matharena_binaryonly_20261020"))
RAW = Path(os.environ.get("PSN_IRT_RAW_DATA", ROOT.parent / "approach1-starter/data/raw"))
ENCODER = Path(os.environ.get("PSN_IRT_ENCODER", ROOT.parent / "approach1-starter/resources/text_encoder"))
BASELINE = ROOT / "results/chunked/experiments/chunked_full_data_mps_20261107_v1"
BASELINE_MANIFEST = (BASELINE / "manifest.json" if (BASELINE / "manifest.json").is_file()
                     else ROOT / "reports/feature_transfer_20261107/data_provenance.json")
SOURCES = ("run_feature_transfer.py", "feature_transfer.py", "models/feature_irt.py",
           "full_data_training.py", "holdout_training.py", "models/text_chunking.py",
           "models/Semantic_IRT.py", "runtime.py")


def plot_results(root):
    import matplotlib.pyplot as plt

    paths = sorted(root.glob("*/item_holdout/training_history.csv"))
    fig, axes = plt.subplots((len(paths)+1)//2, 2, figsize=(13, 3.1*((len(paths)+1)//2)), squeeze=False)
    for ax, path in zip(axes.flat, paths):
        data = pd.read_csv(path)
        for column, label in (("training_row_Brier", "Train (row)"),
                              ("validation_row_Brier", "Validation (row)"),
                              ("validation_equal_pair_Brier", "Validation (equal pair)")):
            ax.plot(data.epoch, data[column], label=label)
        ax.set(title=path.parents[1].name, xlabel="Epoch", ylabel="Brier")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(root / "item_learning_curves.png", dpi=140); plt.close(fig)
    comparison = pd.read_csv(root / "comparison.csv")
    learned = comparison[comparison.predictor == comparison.recipe]
    data = learned[learned.experiment == "item_holdout"].pivot(index="recipe", columns="partition", values="equal_pair_Brier")
    fig, ax = plt.subplots(figsize=(12, 6))
    data[["validation", "heldout"]].plot.bar(ax=ax)
    ax.axhline(.25, ls=":", color="black", label="Constant 0.5")
    ax.set(ylabel="Equal-pair Brier (lower is better)", title="Fixed item holdout — features and training variants")
    ax.legend(); fig.tight_layout(); fig.savefig(root / "item_comparison.png", dpi=150); plt.close(fig)
    outer = learned[(learned.partition == "heldout") & (learned.experiment != "item_holdout")]
    if not outer.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        outer.pivot(index="experiment", columns="recipe", values="equal_pair_Brier").plot.bar(ax=ax)
        ax.axhline(.25, ls=":", color="black", label="Constant 0.5")
        ax.set(ylabel="Equal-pair Brier", title="Repeated benchmark holdouts — checkpoints selected internally")
        ax.legend(); fig.tight_layout(); fig.savefig(root / "benchmark_comparison.png", dpi=150); plt.close(fig)


def run_suite(run_name, settings=None, recipe_names=None, device="auto"):
    torch.set_num_threads(2)
    device = select_device(device)
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    recipes = [r for r in RECIPES if recipe_names is None or r.name in recipe_names]
    if not recipes or (recipe_names and set(recipe_names) != {r.name for r in recipes}):
        raise ValueError("Unknown/empty recipe selection.")
    if Path(run_name).name != run_name:
        raise ValueError("Use a single directory name.")
    for path in (STUDY, RAW, ENCODER, BASELINE_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}. See README data setup.")
    root = ROOT / "results/chunked/experiments" / run_name
    root.mkdir(parents=True, exist_ok=False)
    frame, vectors, audit = load_features(STUDY, RAW, ENCODER, BASELINE_MANIFEST)
    source_hashes = {}
    for name in SOURCES:
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        source_hashes[name] = sha(destination)
    manifest = dict(model_version=MODEL_VERSION, started_utc=datetime.now(timezone.utc).isoformat(),
                    data_audit=audit, source_sha256=source_hashes,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    settings=settings, recipes=[asdict(r) for r in recipes],
                    packages={p: importlib.metadata.version(p) for p in
                              ("torch", "pandas", "numpy", "sentence-transformers", "transformers")},
                    baseline_manifest_sha256=sha(BASELINE_MANIFEST),
                    recorded_splits=json.loads((STUDY / "prepared.json").read_text())["splits"],
                    device=device,
                    selection="Choose recipe by fixed item internal-validation equal-pair Brier before scoring any outer tests.",
                    benchmark_policy="Evaluate ordered-name control, full metadata and validation-selected recipe; duplicates removed.",
                    limitations="Previously observed outer tests are repeated evaluation sets; not fresh confirmation. One seed. No hosted submission.")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    identity_audit(frame).to_csv(root / "identity_audit.csv", index=False)
    item_split = recorded_split(frame, STUDY, "item")
    for recipe in recipes:
        run_fold(item_split, vectors, root / recipe.name / "item_holdout", settings, recipe, "item_holdout", device=device)
    # Lock recipe choice BEFORE scoring any outer item responses.
    validation = []
    for recipe in recipes:
        directory = root / recipe.name / "item_holdout"
        table = pd.read_csv(directory / "comparison.csv")
        value = table[(table.partition == "validation") & (table.predictor == recipe.name)].iloc[0]
        validation.append(dict(recipe=recipe.name, equal_pair_Brier=float(value.equal_pair_Brier),
                               checkpoint_sha256=sha(directory / "best_model.pt")))
    selected = min(validation, key=lambda row: row["equal_pair_Brier"])["recipe"]
    (root / "selection.json").write_text(json.dumps(dict(
        selected_recipe=selected, candidates=validation, locked_utc=datetime.now(timezone.utc).isoformat(),
        selection_metric="item internal-validation equal-pair Brier", outer_test_used=False), indent=2))
    print(f"Validation-selected recipe locked: {selected}", flush=True)
    for recipe in recipes:
        directory = root / recipe.name / "item_holdout"
        model, encoder, _, checkpoint = load_checkpoint(directory / "best_model.pt", device)
        evaluate_fold(model, encoder, recipe, item_split, vectors, directory, "item_holdout", checkpoint["best_epoch"], device)
        metadata = json.loads((directory / "settings.json").read_text())
        metadata["outer_scored"] = True
        (directory / "settings.json").write_text(json.dumps(metadata, indent=2))
        del model
        clear_cache(device)
    benchmark_recipes = [r for r in recipes if r.name in {"ordered_name", "full_metadata", selected}]
    for name in sorted(manifest["recorded_splits"]):
        if not name.startswith("benchmark__"):
            continue
        split = recorded_split(frame, STUDY, name)
        for recipe in benchmark_recipes:
            run_fold(split, vectors, root / recipe.name / name, settings, recipe, name, device=device, score_outer=True)
    tables = [pd.read_csv(p) for p in sorted(root.glob("*/*/comparison.csv"))]
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(root / "comparison.csv", index=False)
    learned = comparison[comparison.predictor == comparison.recipe]
    summary = learned[(learned.partition == "heldout") & (learned.experiment != "item_holdout")].groupby("recipe").equal_pair_Brier.agg(["mean", "count"])
    summary.to_csv(root / "benchmark_macro.csv")
    plot_results(root)
    for name, expected in source_hashes.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Training code changed mid-run: {name}")
    (root / "complete.json").write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        selected_recipe=selected, trained_folds=len(tables), comparison_sha256=sha(root / "comparison.csv")), indent=2))
    print(f"Completed {len(tables)} fits on {device}: {root}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--recipes", nargs="+")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5814)
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, cuda:0, or mps")
    args = parser.parse_args()
    run_suite(args.run_name, dict(epochs=args.epochs, seed=args.seed), args.recipes, args.device)
