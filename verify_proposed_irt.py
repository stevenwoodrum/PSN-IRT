"""Independent source/split/feature/score/export audit of a proposed-2PL run."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from full_data_training import sha
from models.proposed_irt import A1_SUBJECT_FIELDS, ALL_SUBJECT_FIELDS, item_feature_tokens
from proposed_training import load_checkpoint, make_data, metrics, predict
from run_feature_transfer import BASELINE_MANIFEST, RAW, STUDY
from runtime import clear_cache, select_device


def raw_truth():
    """Rebuild row identities, labels and all predict() fields directly from source files."""
    truth = pd.read_parquet(STUDY / "observations.parquet", columns=[
        "row_id", "dataset", "subject_id", "item_id", "response_id", "response", "item_group", "item_content"]
    ).rename(columns={"dataset": "benchmark_id"})
    parts = []
    for benchmark, part in truth.groupby("benchmark_id", sort=False):
        subjects = pd.read_parquet(RAW / benchmark / "subjects.parquet").set_index("subject_id")
        items = pd.read_parquet(RAW / benchmark / "items.parquet", columns=["item_id", "item_features"]).set_index("item_id")
        part = part.copy()
        for field in ALL_SUBJECT_FIELDS:
            part[field] = subjects.loc[part.subject_id, field].fillna("").astype(str).to_numpy()
        part["item_features"] = items.loc[part.item_id, "item_features"].fillna("").astype(str).to_numpy()
        parts.append(part)
    truth = pd.concat(parts).sort_values("row_id").set_index("row_id", drop=False)
    assert len(truth) == 229143 and np.array_equal(truth.row_id, np.arange(len(truth)))
    return truth


def verify(root, device="auto", prediction_tolerance=1e-6):
    torch.set_num_threads(2)
    device = select_device(device)
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert sha(BASELINE_MANIFEST) == manifest["baseline_manifest_sha256"]
    for file, expected in manifest["source_sha256"].items():
        assert sha(root / "source" / file) == expected, file
    for file, expected in manifest["data_audit"]["source_sha256"].items():
        assert sha(STUDY / file) == expected, file
    truth = raw_truth()
    prompts = json.loads((STUDY / "prompts.json").read_text())
    vectors = np.load(STUDY / "prompt_vectors.npy", allow_pickle=False)
    by_text = dict(zip(prompts, vectors))
    contents = truth.drop_duplicates("item_id")
    item_vectors = {i: by_text[t] for i, t in zip(contents.item_id, contents.item_content)}
    identity = pd.DataFrame(manifest["data_audit"]["identity_by_benchmark"]).set_index("benchmark")
    for benchmark, part in truth.groupby("benchmark_id"):
        assert identity.loc[benchmark, "all_field_tuples"] == len(part[list(ALL_SUBJECT_FIELDS)].drop_duplicates())
    selection = json.loads((root / "selection.json").read_text())
    assert selection["outer_test_used"] is False
    candidates = selection["candidates"]
    assert selection["selected_recipe"] == min(candidates, key=lambda c: c["equal_pair_Brier"])["recipe"]
    for candidate in candidates:
        assert sha(root / candidate["recipe"] / "item_holdout/best_model.pt") == candidate["checkpoint_sha256"]
    checks, hashes, tables = [], {}, []
    max_pred, max_export, max_metric, max_loose = 0., 0., 0., {}
    string_columns = {f: str for f in ALL_SUBJECT_FIELDS + ("item_features", "subject_id", "item_id", "response_id", "item_group")}
    for checkpoint_path in sorted(root.glob("*/*/best_model.pt")):
        directory = checkpoint_path.parent
        recipe_name, experiment = directory.parent.name, directory.name
        name = "item" if experiment == "item_holdout" else experiment
        split_source = STUDY / "splits" / name / "membership.parquet"
        assert sha(split_source) == manifest["recorded_splits"][name]["partition_sha256"]
        membership = pd.read_parquet(split_source).set_index("row_id")
        saved = pd.read_csv(directory / "predictions.csv", keep_default_na=False, dtype=string_columns)
        assert len(saved) == len(truth) and saved.row_id.nunique() == len(truth)
        expected_rows = truth.loc[saved.row_id]
        for field in ("benchmark_id", "subject_id", "item_id", "response_id", "response", "item_group",
                      "item_features") + ALL_SUBJECT_FIELDS:
            assert np.array_equal(saved[field].to_numpy(), expected_rows[field].to_numpy()), (directory, field)
        expected_part = membership.loc[saved.row_id, "partition"].replace({"test": "heldout"}).to_numpy()
        assert np.array_equal(saved.partition, expected_part), directory
        train, validation, outer = (saved[saved.partition == k] for k in ("train", "validation", "heldout"))
        for left, right in ((train, validation), (train, outer), (validation, outer)):
            assert set(left.item_group).isdisjoint(right.item_group) and set(left.item_id).isdisjoint(right.item_id)
        if experiment.startswith("benchmark__"):
            benchmark = experiment.removeprefix("benchmark__")
            assert set(outer.benchmark_id) == {benchmark}
            assert benchmark not in set(train.benchmark_id) | set(validation.benchmark_id)
        settings = json.loads((directory / "settings.json").read_text())
        training_device = settings["training_device"]
        assert settings["training_tensor_devices"] == [training_device]
        if torch.device(training_device).type != "cpu":
            assert settings["accelerator_allocated_bytes_before_training"] > 0
        assert settings["outer_scored"] is True
        model, subject, tokens, recipe, checkpoint = load_checkpoint(checkpoint_path, device)
        assert recipe.name == recipe_name
        # Train-only categories and vocabulary; never IDs.
        for field in recipe.fields:
            assert subject.categories[field] == sorted(train[field].unique()), (directory, field)
        assert set(subject.categories) == set(recipe.fields) and set(recipe.fields) <= set(ALL_SUBJECT_FIELDS)
        if recipe.item_features:
            vocabulary = sorted({t for s in train.drop_duplicates("item_id").item_features for t in item_feature_tokens(s)})
            assert tokens.vocabulary == vocabulary, directory
        else:
            assert tokens is None and checkpoint["item_vocabulary"] == []
        assert np.array_equal(saved.known_subject, saved.normalized_name.isin(set(train.normalized_name)))
        seen = set(train[list(ALL_SUBJECT_FIELDS)].itertuples(index=False, name=None))
        assert np.array_equal(saved.known_subject_record, [t in seen for t in saved[list(ALL_SUBJECT_FIELDS)].itertuples(index=False, name=None)])
        p = predict(model, make_data(saved, item_vectors, subject, tokens, recipe, device))
        error = float(np.max(abs(p - saved.probability.to_numpy())))
        assert error < prediction_tolerance, (directory, error)
        max_pred = max(max_pred, error)
        params = pd.read_csv(directory / "item_parameters.csv", keep_default_na=False, dtype={"item_id": str}).set_index("item_id")
        assert params.index.is_unique and set(params.index) == set(saved.item_id) and (params.a > 0).all()
        ability = pd.read_csv(directory / "subject_abilities.csv", keep_default_na=False, dtype={f: str for f in ALL_SUBJECT_FIELDS})
        joined = saved.merge(ability, on=list(ALL_SUBJECT_FIELDS), validate="many_to_one")
        assert len(joined) == len(saved)
        b, a = params.loc[joined.item_id, "b"].to_numpy(), params.loc[joined.item_id, "a"].to_numpy()
        reconstructed = 1 / (1 + np.exp(np.clip(-a * (joined.ability.to_numpy() - b), -700, 700)))
        export_error = float(np.max(abs(reconstructed - joined.probability.to_numpy())))
        assert export_error < 1e-6, (directory, export_error)
        max_export = max(max_export, export_error)
        comparison = pd.read_csv(directory / "comparison.csv")
        for record in comparison.to_dict("records"):
            part = saved[saved.partition == record["partition"]]
            values = (part.probability.to_numpy() if record["predictor"] == recipe_name else
                      np.full(len(part), .5 if record["predictor"] == "constant 0.5" else train.response.mean()))
            for metric, value in metrics(part, values).items():
                if metric == "auc_pairs":
                    assert value == record[metric]
                    continue
                if np.isnan(value):
                    assert np.isnan(record[metric]); continue
                difference = abs(value - record[metric])
                # Saved CSV probabilities carry ~8 significant digits of float32.
                # Brier/ECE move by <1e-7. BCE's log term amplifies that rounding
                # for BCE-trained predictions very near 0/1, and rounding can create
                # or break AUC ties, so those get explicit looser tolerances.
                # Checkpoint reproduction above is the strict prediction check.
                # (Near the 1e-7 clip, 1/(1-p) ~ 1e7, so 8-digit CSV rounding can
                # move a few saturated rows' log loss noticeably: allow 0.1% relative.)
                tolerance = {"BCE": 1e-3 * max(1., abs(value)), "within_pair_AUC": 1e-4}.get(metric, 1e-7)
                assert difference < tolerance, (directory, metric, difference)
                max_metric = max(max_metric, difference) if metric not in ("BCE", "within_pair_AUC") else max_metric
                max_loose[metric] = max(max_loose.get(metric, 0.), difference)
        history = pd.read_csv(directory / "training_history.csv")
        best = int(history.loc[history.validation_equal_pair_Brier.idxmin(), "epoch"])
        assert best == checkpoint["best_epoch"] == settings["best_epoch"]
        tables.append(comparison)
        for file in directory.iterdir():
            if file.is_file():
                hashes[str(file.relative_to(root))] = sha(file)
        checks.append(dict(recipe=recipe_name, experiment=experiment, rows=len(saved), best_epoch=best,
                           max_prediction_error=error, max_export_error=export_error,
                           training_device=training_device, audit_device=str(next(model.parameters()).device)))
        print(f"Verified {recipe_name}/{experiment}", flush=True)
        del model
        clear_cache(device)
    aggregate = pd.concat(tables, ignore_index=True)
    pd.testing.assert_frame_equal(aggregate.reset_index(drop=True), pd.read_csv(root / "comparison.csv"))
    complete = json.loads((root / "complete.json").read_text())
    assert complete["trained_folds"] == len(checks) and complete["comparison_sha256"] == sha(root / "comparison.csv")
    receipt = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
                   verifier_sha256=sha(Path(__file__)), folds=checks, raw_feature_rejoin=True,
                   exact_recorded_membership=True, train_only_categories_and_vocabulary=True,
                   all_subjects_retained=True, audit_device=device, prediction_tolerance=prediction_tolerance,
                   max_prediction_error=max_pred, max_export_error=max_export, max_brier_ece_metric_error=max_metric,
                   max_csv_recompute_difference_by_metric=max_loose,
                   artifact_sha256=hashes,
                   limitation="Repeated outer tests; not prospective model selection or hosted performance.")
    output = root / "VERIFICATION.json"
    if output.exists():
        output = root / f"VERIFICATION_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.write_text(json.dumps(receipt, indent=2))
    print(f"PASS: {len(checks)} fits; max prediction error {max_pred:.3g}; export error {max_export:.3g}; metric error {max_metric:.3g}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    verify(args.run_directory, args.device, args.prediction_tolerance)
