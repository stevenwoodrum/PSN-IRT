"""Independent source/split/score/export audit of a feature-transfer run."""

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

from feature_transfer import load_checkpoint, make_tensors, predict
from full_data_training import sha
from models.feature_irt import SUBJECT_FIELDS
from run_feature_transfer import BASELINE_MANIFEST, STUDY
from runtime import clear_cache, select_device


def metrics(table, p):
    y = table.response.to_numpy(dtype=float)
    error = pd.DataFrame(dict(benchmark=table.benchmark_id, subject=table.subject_id, error=(y-p)**2))
    clipped = np.clip(p, 1e-7, 1-1e-7)
    return dict(row_Brier=float(error.error.mean()),
                equal_pair_Brier=float(error.groupby(["benchmark", "subject"]).error.mean().mean()),
                BCE=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()))


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
    truth = pd.read_parquet(STUDY / "observations.parquet", columns=[
        "row_id", "dataset", "subject_id", "item_id", "response_id", "response", "item_group",
        "subject_normalized_name", "subject_provider", "subject_reasoning_effort"]).rename(columns={
        "dataset": "benchmark_id", **{"subject_"+f: f for f in SUBJECT_FIELDS}})
    for field in SUBJECT_FIELDS:
        truth[field] = truth[field].fillna("").astype(str)
    truth = truth.set_index("row_id", drop=False)
    assert len(truth) == 229143
    prompts = json.loads((STUDY / "prompts.json").read_text())
    vectors = np.load(STUDY / "prompt_vectors.npy", allow_pickle=False)
    # Reuse only the verified prompt/vector mapping; item identity is joined
    # directly from the prepared source rather than the generated predictions.
    contents = pd.read_parquet(STUDY / "observations.parquet", columns=["item_id", "item_content"]).drop_duplicates("item_id")
    by_text = dict(zip(prompts, vectors))
    item_vectors = {item: by_text[text] for item, text in contents.itertuples(index=False, name=None)}
    checks, hashes = [], {}
    max_prediction_error, max_export_error, max_metric_error = 0., 0., 0.
    selection = json.loads((root / "selection.json").read_text())
    assert selection["outer_test_used"] is False
    candidates = selection["candidates"]
    assert selection["selected_recipe"] == min(candidates, key=lambda c: c["equal_pair_Brier"])["recipe"]
    for candidate in candidates:
        checkpoint = root / candidate["recipe"] / "item_holdout/best_model.pt"
        assert sha(checkpoint) == candidate["checkpoint_sha256"]
    tables = []
    for checkpoint_path in sorted(root.glob("*/*/best_model.pt")):
        directory = checkpoint_path.parent
        recipe_name, experiment = directory.parent.name, directory.name
        name = "item" if experiment == "item_holdout" else experiment
        split_source = STUDY / "splits" / name / "membership.parquet"
        assert sha(split_source) == manifest["recorded_splits"][name]["partition_sha256"]
        membership = pd.read_parquet(split_source).set_index("row_id")
        saved = pd.read_csv(directory / "predictions.csv", keep_default_na=False)
        assert len(saved) == len(truth) and saved.row_id.nunique() == len(truth)
        for field in truth.columns:
            assert np.array_equal(saved[field], truth.loc[saved.row_id, field]), (directory, field)
        expected_part = membership.loc[saved.row_id, "partition"].replace({"test": "heldout"}).to_numpy()
        assert np.array_equal(saved.partition, expected_part), directory
        train = saved[saved.partition == "train"]
        validation = saved[saved.partition == "validation"]
        outer = saved[saved.partition == "heldout"]
        for left, right in ((train, validation), (train, outer), (validation, outer)):
            assert set(left.item_group).isdisjoint(right.item_group)
            assert set(left.item_id).isdisjoint(right.item_id)
        if experiment.startswith("benchmark__"):
            benchmark = experiment.removeprefix("benchmark__")
            assert set(outer.benchmark_id) == {benchmark}
            assert benchmark not in set(train.benchmark_id) | set(validation.benchmark_id)
        settings = json.loads((directory / "settings.json").read_text())
        training_device = settings["training_device"]
        assert settings["training_tensor_devices"] == [training_device]
        if torch.device(training_device).type != "cpu":
            allocation = settings.get("accelerator_allocated_bytes_before_training",
                                      settings.get("mps_allocated_bytes_before_training"))
            assert allocation > 0
        assert settings["outer_scored"] is True
        model, encoder, recipe, checkpoint = load_checkpoint(checkpoint_path, device)
        assert recipe.name == recipe_name
        for field in recipe.fields:
            expected = (list(dict.fromkeys(train[field])) if recipe.subject_mode == "name_mean"
                        else sorted(train[field].unique()))
            assert encoder.categories[field] == expected
        actual_known = saved.normalized_name.isin(set(train.normalized_name))
        assert np.array_equal(saved.known_subject, actual_known)
        probabilities = predict(model, make_tensors(saved, item_vectors, encoder, recipe, device))
        error = float(np.max(abs(probabilities - saved.probability.to_numpy())))
        assert error < prediction_tolerance, (directory, error, prediction_tolerance)
        max_prediction_error = max(max_prediction_error, error)
        params = pd.read_csv(directory / "item_parameters.csv", keep_default_na=False).set_index("item_id")
        assert params.index.is_unique and set(params.index) == set(saved.item_id)
        assert (params.a > 0).all() and (params.c >= 0).all() and (params.d >= params.c).all() and (params.d <= 1).all()
        ability = pd.read_csv(directory / "subject_abilities.csv", keep_default_na=False)
        joined = saved.merge(ability, on=list(SUBJECT_FIELDS), validate="many_to_one")
        beta, a, c, d = [params.loc[joined.item_id, field].to_numpy() for field in ("beta", "a", "c", "d")]
        reconstructed = c+(d-c)/(1+np.exp(np.clip(-a*(joined.ability.to_numpy()-beta), -700, 700)))
        export_error = float(np.max(abs(reconstructed-joined.probability.to_numpy())))
        assert export_error < 1e-6
        max_export_error = max(max_export_error, export_error)
        comparison = pd.read_csv(directory / "comparison.csv")
        for record in comparison.to_dict("records"):
            part = saved[saved.partition == record["partition"]]
            p = (part.probability.to_numpy() if record["predictor"] == recipe_name else
                 np.full(len(part), .5 if record["predictor"] == "constant 0.5" else train.response.mean()))
            for metric, value in metrics(part, p).items():
                difference = abs(value-record[metric])
                # pandas emits float32 prediction columns at float32 decimal
                # precision. Re-reading them as float64 can move a recomputed
                # metric by a few 1e-10; checkpoint reproduction is checked
                # separately above, rather than requiring impossible exact CSV
                # reconstruction of every binary float.
                assert difference < 1e-8, (directory, metric, difference)
                max_metric_error = max(max_metric_error, difference)
        history = pd.read_csv(directory / "training_history.csv")
        best = int(history.loc[history.validation_equal_pair_Brier.idxmin(), "epoch"])
        assert best == checkpoint["best_epoch"] == settings["best_epoch"]
        if experiment == "item_holdout":
            candidate = next(c for c in candidates if c["recipe"] == recipe_name)
            val_score = comparison[(comparison.partition == "validation") & (comparison.predictor == recipe_name)].equal_pair_Brier.iloc[0]
            assert abs(candidate["equal_pair_Brier"]-val_score) < 1e-12
        tables.append(comparison)
        for file in directory.iterdir():
            if file.is_file():
                hashes[str(file.relative_to(root))] = sha(file)
        checks.append(dict(recipe=recipe_name, experiment=experiment, rows=len(saved),
                           best_epoch=best, max_prediction_error=error, max_export_error=export_error,
                           ordering_violations=0, training_device=training_device,
                           audit_device=str(next(model.parameters()).device)))
        print(f"Verified {recipe_name}/{experiment}: {len(saved):,} rows", flush=True)
        del model
        clear_cache(device)
    aggregate = pd.concat(tables, ignore_index=True)
    published = pd.read_csv(root / "comparison.csv")
    pd.testing.assert_frame_equal(aggregate.reset_index(drop=True), published)
    complete = json.loads((root / "complete.json").read_text())
    assert complete["trained_folds"] == len(checks)
    assert complete["comparison_sha256"] == sha(root / "comparison.csv")
    receipt = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)), folds=checks, source_and_cohort_hashes=True,
        exact_recorded_membership=True, train_only_categories=True, all_subjects_retained=True,
        audit_device=device, prediction_tolerance=prediction_tolerance,
        source_parameters_reconstruct_predictions=True,
        max_prediction_error=max_prediction_error, max_export_error=max_export_error,
        max_metric_error=max_metric_error, artifact_sha256=hashes,
        limitation="One seed; repeated outer tests; not independent prospective model selection or hosted performance.")
    output = root / "VERIFICATION.json"
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / f"VERIFICATION_{stamp}.json"
    output.write_text(json.dumps(receipt, indent=2))
    print(f"PASS: {len(checks)} fits; max prediction error {max_prediction_error:.3g}; export error {max_export_error:.3g}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    verify(args.run_directory, args.device, args.prediction_tolerance)
