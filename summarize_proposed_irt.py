"""Compare proposed-2PL runs with historical baselines using one metric implementation.

Reads saved predictions only (no training). Usage:
    python summarize_proposed_irt.py RUN_DIR [RUN_DIR ...] --out OUTPUT_DIR
Several RUN_DIRs (different seeds) are summarized as mean/sd/min/max per recipe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from proposed_training import metrics
from run_feature_transfer import ROOT, STUDY

EXPERIMENTS = ROOT / "results/chunked/experiments"
BENCHMARKS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
FOLDS = ("item_holdout",) + tuple("benchmark__" + b for b in BENCHMARKS)


def a2_like(path):
    table = pd.read_csv(path, usecols=["benchmark_id", "subject_id", "response", "partition", "probability"],
                        dtype={"subject_id": str})
    return table


def a1(method, fold):
    name = "item" if fold == "item_holdout" else fold
    base = STUDY / "results" / method / name / "seed_5814"
    parts = []
    for role, file in (("validation", "validation_predictions.parquet"), ("heldout", "test_predictions.parquet")):
        t = pd.read_parquet(base / file).rename(columns={"dataset": "benchmark_id"})
        parts.append(t.assign(partition=role)[["benchmark_id", "subject_id", "response", "partition", "probability"]])
    return pd.concat(parts)


def baselines():
    return {
        "A2 historical Semantic IRT (4PL, MSE, name)": lambda f: a2_like(EXPERIMENTS / "chunked_full_data_mps_20261107_v1" / f / "predictions.csv"),
        "A2 feature transfer full_metadata (ordered 4PL, MSE)": lambda f: a2_like(EXPERIMENTS / "feature_transfer_mps_20261107_v1/full_metadata" / f / "predictions.csv"),
        "A1 M03 original FNN (BCE)": lambda f: a1("M03", f),
    }


def score_rows(label, loader, kind):
    rows = []
    for fold in FOLDS:
        table = loader(fold)
        for partition in ("validation", "heldout"):
            part = table[table.partition == partition]
            p = np.full(len(part), .5) if kind == "constant" else part.probability.to_numpy()
            rows.append(dict(model=label, fold=fold, partition=partition, scope="all", rows=len(part), **metrics(part, p)))
            if fold == "item_holdout" and partition == "heldout":
                for benchmark, sub in part.groupby("benchmark_id"):
                    q = np.full(len(sub), .5) if kind == "constant" else sub.probability.to_numpy()
                    rows.append(dict(model=label, fold=fold, partition=partition, scope=benchmark, rows=len(sub), **metrics(sub, q)))
    return rows


def run_rows(run):
    rows = []
    seed = json.loads((run / "manifest.json").read_text())["settings"]["seed"]
    for recipe_dir in sorted(p for p in run.iterdir() if (p / "item_holdout").is_dir()):
        for fold in FOLDS:
            directory = recipe_dir / fold
            settings = json.loads((directory / "settings.json").read_text())
            table = pd.read_csv(directory / "predictions.csv", usecols=["benchmark_id", "subject_id", "response", "partition",
                                                                          "probability", "known_subject", "known_subject_record"],
                                dtype={"subject_id": str})
            for partition in ("validation", "heldout"):
                part = table[table.partition == partition]
                base = dict(model=recipe_dir.name, seed=seed, fold=fold, partition=partition, best_epoch=settings["best_epoch"],
                            unseen_name_rows=int((~part.known_subject).sum()),
                            unseen_record_rows=int((~part.known_subject_record).sum()))
                rows.append(dict(**base, scope="all", rows=len(part), **metrics(part, part.probability.to_numpy())))
                if fold == "item_holdout" and partition == "heldout":
                    for benchmark, sub in part.groupby("benchmark_id"):
                        rows.append(dict(**base, scope=benchmark, rows=len(sub), **metrics(sub, sub.probability.to_numpy())))
    return rows


def headline(frame):
    allrows = frame[frame.scope == "all"]
    item_val = allrows[(allrows.fold == "item_holdout") & (allrows.partition == "validation")].set_index("model")
    item_out = allrows[(allrows.fold == "item_holdout") & (allrows.partition == "heldout")].set_index("model")
    bench = allrows[(allrows.fold != "item_holdout") & (allrows.partition == "heldout")]
    macro = bench.groupby("model")[["equal_pair_Brier", "row_Brier", "within_pair_AUC"]].mean()
    out = pd.DataFrame({
        "item_validation_pair_Brier": item_val.equal_pair_Brier,
        "item_outer_pair_Brier": item_out.equal_pair_Brier,
        "item_outer_row_Brier": item_out.row_Brier,
        "item_outer_within_pair_AUC": item_out.within_pair_AUC,
        "item_outer_row_ECE": item_out.row_ECE,
        "benchmark_macro_pair_Brier": macro.equal_pair_Brier,
        "benchmark_macro_row_Brier": macro.row_Brier,
        "benchmark_macro_within_pair_AUC": macro.within_pair_AUC,
    })
    if "best_epoch" in item_out:
        out["item_best_epoch"] = item_out.best_epoch
    return out


def main(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    reference = []
    for label, loader in baselines().items():
        reference += score_rows(label, loader, "model")
    reference += score_rows("Constant 0.5", lambda f: a2_like(EXPERIMENTS / "chunked_full_data_mps_20261107_v1" / f / "predictions.csv"), "constant")
    reference = pd.DataFrame(reference)
    proposed = pd.DataFrame([r for run in runs for r in run_rows(run)])
    reference.to_csv(output / "baseline_scores.csv", index=False)
    proposed.to_csv(output / "proposed_scores.csv", index=False)
    per_seed = []
    for seed, group in proposed.groupby("seed"):
        per_seed.append(headline(group).assign(seed=seed))
    per_seed = pd.concat(per_seed).reset_index().rename(columns={"index": "model"})
    per_seed.to_csv(output / "proposed_headline_by_seed.csv", index=False)
    numeric = per_seed.drop(columns="seed").groupby("model")
    summary = numeric.mean().add_suffix("_mean")
    if per_seed.seed.nunique() > 1:
        summary = summary.join(numeric.std().add_suffix("_sd")).join(numeric.min().add_suffix("_min")).join(numeric.max().add_suffix("_max"))
    summary["seeds"] = per_seed.groupby("model").seed.nunique()
    summary.to_csv(output / "proposed_headline_summary.csv")
    headline(reference).to_csv(output / "baseline_headline.csv")
    item_bench = pd.concat([reference, proposed])
    item_bench = item_bench[(item_bench.fold == "item_holdout") & (item_bench.partition == "heldout") & (item_bench.scope != "all")]
    item_bench.groupby(["model", "scope"])[["equal_pair_Brier", "within_pair_AUC"]].mean().unstack("scope").to_csv(output / "item_outer_by_benchmark.csv")
    bench = pd.concat([reference, proposed])
    bench = bench[(bench.fold != "item_holdout") & (bench.partition == "heldout") & (bench.scope == "all")]
    bench.groupby(["model", "fold"])[["equal_pair_Brier", "within_pair_AUC", "row_Brier"]].mean().unstack("fold").to_csv(output / "benchmark_outer_by_fold.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(headline(reference).round(4))
        print(summary.round(4))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.runs, args.out)
