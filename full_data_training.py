"""Read the full Approach 1 cohort, verified chunk cache and recorded splits."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import text_chunking


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_data(study, raw_root, encoder_root):
    study, raw_root, encoder_root = map(Path, (study, raw_root, encoder_root))
    prepared = json.loads((study / "prepared.json").read_text())
    for name, expected in prepared["source_files"].items():
        path = raw_root.joinpath(*Path(name).parts[2:])
        if sha(path) != expected:
            raise ValueError(f"Raw source changed since Approach 1: {path}")
    columns = ["dataset", "subject_id", "item_id", "response_id", "response",
               "subject_normalized_name", "item_group", "row_id"]
    frame = pd.read_parquet(study / "observations.parquet", columns=columns).rename(
        columns={"dataset": "benchmark_id", "subject_normalized_name": "normalized_name"})
    if len(frame) != prepared["eligible_rows"] or not frame.response.isin([0, 1]).all():
        raise ValueError("Prepared cohort size or binary labels differ from the recorded population.")
    assert np.array_equal(frame.row_id, np.arange(len(frame)))
    assert not frame.duplicated(["benchmark_id", "response_id"]).any()

    contents, inventory = {}, []
    for folder in sorted(raw_root.iterdir()):
        if not (folder / "response.parquet").is_file():
            continue
        response = pd.read_parquet(folder / "response.parquet")
        retained = frame[frame.benchmark_id == folder.name]
        inventory.append({"benchmark": folder.name, "downloaded_rows": len(response), "eligible_rows": len(retained)})
        if retained.empty:
            continue
        expected = response[response.response.isin([0, 1])]
        assert set(expected.response_id) == set(retained.response_id)
        original = expected.set_index("response_id").loc[retained.response_id]
        for column in ("subject_id", "item_id", "response"):
            assert np.array_equal(original[column], retained[column]), (folder.name, column)
        subjects = pd.read_parquet(folder / "subjects.parquet").set_index("subject_id")
        assert np.array_equal(subjects.loc[retained.subject_id, "normalized_name"].fillna(""), retained.normalized_name)
        items = pd.read_parquet(folder / "items.parquet", columns=["item_id", "content"])
        for item, text in items[items.item_id.isin(retained.item_id)].itertuples(index=False, name=None):
            text = "" if pd.isna(text) else str(text)
            if item in contents and contents[item] != text:
                raise ValueError("The same item ID has conflicting prompt text.")
            contents[item] = text
    # Object dtype shares each unique text rather than expanding long papers per response.
    frame["content"] = pd.Series([contents[item] for item in frame.item_id], dtype=object)

    frozen_source = study / "source/frozen_features.py"
    if sha(frozen_source) != prepared["source_code"]["frozen_features.py"]:
        raise ValueError("Approach 1's saved embedding implementation changed.")
    old_tree = ast.parse(frozen_source.read_text())
    new_tree = ast.parse(Path(text_chunking.__file__).read_text())
    for name in ("make_text_encoder", "prepare_chunk_features", "prompt_chunks", "embed_text"):
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if name == "make_text_encoder":
            old.body = old.body[1:]  # The port's loader also accepts a Hub name instead of requiring a local directory.
        assert ast.dump(old) == ast.dump(new), f"Embedding implementation differs: {name}"

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    features = module("approach1_frozen_features", frozen_source)
    cache = module("approach1_prompt_cache", study.parents[1] / "prompt_embedding_cache.py")
    functions = (features.make_text_encoder, features.prepare_chunk_features, features.prompt_chunks, features.embed_text)
    settings = {"max_length": 2048, "chunk_size": 512, "pooling": "token_weighted_mean_l2", "dtype": "float32"}
    prompts = json.loads((study / "prompts.json").read_text())
    vectors = np.load(study / "prompt_vectors.npy", allow_pickle=False)
    reuse = json.loads((study / "audits/embedding_reuse.json").read_text())
    assert sha(study / "prompt_vectors.npy") == reuse["combined_vectors_sha256"]
    original_study = study.parent / "approach1_comparison_20261020"
    old_prompts = json.loads((original_study / "prompts.json").read_text())
    old_lookup = {text: index for index, text in enumerate(old_prompts)}
    missing = [text for text in prompts if text not in old_lookup]
    old_key = cache._cache_key(encoder_root, cache._prompt_hashes(old_prompts), settings, functions)
    new_key = cache._cache_key(encoder_root, cache._prompt_hashes(missing), settings, functions)
    with np.load(original_study / "embedding_cache" / (old_key + ".npz"), allow_pickle=False) as old_cache, \
            np.load(study / "embedding_cache" / (new_key + ".npz"), allow_pickle=False) as new_cache:
        assert old_cache["key"].item() == old_key == reuse["original_cache_key"]
        assert new_cache["key"].item() == new_key
        assert np.array_equal(old_cache["prompt_hashes"], cache._prompt_hashes(old_prompts))
        assert np.array_equal(new_cache["prompt_hashes"], cache._prompt_hashes(missing))
        fresh_lookup = {text: index for index, text in enumerate(missing)}
        old_vectors, new_vectors = old_cache["vectors"], new_cache["vectors"]
        expected_vectors = np.asarray([old_vectors[old_lookup[text]] if text in old_lookup else
                                       new_vectors[fresh_lookup[text]] for text in prompts])
        assert np.array_equal(vectors, expected_vectors)
    assert vectors.shape == (len(prompts), 384) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    by_text = dict(zip(prompts, vectors))
    item_vectors = {item: by_text[text] for item, text in contents.items()}
    audit = {"inventory": inventory, "downloaded_rows": sum(row["downloaded_rows"] for row in inventory),
             "eligible_rows": len(frame), "distinct_items": frame.item_id.nunique(), "unique_prompts": len(prompts),
             "original_cache_key": old_key, "added_matharena_cache_key": new_key,
             "cache_reuse": "Exact vectors verified against cache keys recomputed from current encoder files, libraries and matching preprocessing code.",
             "source_sha256": {name: sha(study / name) for name in
                               ("prepared.json", "observations.parquet", "prompts.json", "prompt_vectors.npy")},
             "eligibility": "Same as latest Approach 1: four binary benchmarks plus MathArena exact numeric 0/1 endpoints; no rounding; MMDocRAG and 83 fractional MathArena rows excluded."}
    print(f"Verified full cohort: {len(frame):,} responses, {len(prompts):,} cached prompt vectors.", flush=True)
    return frame, item_vectors, audit


def recorded_split(frame, study, name):
    study = Path(study)
    record = json.loads((study / "prepared.json").read_text())["splits"][name]
    folder = study / "splits" / name
    if sha(folder / "membership.parquet") != record["partition_sha256"]:
        raise ValueError("Approach 1 split membership changed.")
    membership = pd.read_parquet(folder / "membership.parquet")
    for column in ("row_id", "subject_id", "item_id", "response_id", "item_group"):
        assert np.array_equal(membership[column], frame[column]), column
    assert np.array_equal(membership.dataset, frame.benchmark_id)
    with np.load(folder / "indices.npz", allow_pickle=False) as indices:
        parts = {role: frame.iloc[indices[key]].copy() for role, key in
                 (("train", "train"), ("validation", "validation"), ("heldout", "test"))}
        for role, key in (("train", "train"), ("validation", "validation"), ("heldout", "test")):
            assert np.array_equal(np.flatnonzero(membership.partition == key), indices[key])
    parts["purged"] = frame[membership.partition == "excluded_duplicate_group"].copy()
    assert sum(map(len, parts.values())) == len(frame)
    for i, left in enumerate(parts):
        for right in list(parts)[i + 1:]:
            assert set(parts[left].row_id).isdisjoint(parts[right].row_id)
            if "purged" not in (left, right):
                assert set(parts[left].item_group).isdisjoint(parts[right].item_group)
                assert set(parts[left].item_id).isdisjoint(parts[right].item_id)
    if name.startswith("benchmark__"):
        benchmark = name.removeprefix("benchmark__")
        assert set(parts["heldout"].benchmark_id) == {benchmark}
        assert benchmark not in set(parts["train"].benchmark_id) | set(parts["validation"].benchmark_id)
    return parts
