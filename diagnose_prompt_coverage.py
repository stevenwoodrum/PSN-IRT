"""Full-cohort tokenizer coverage; descriptive only, no accuracy inference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from full_data_training import sha
from run_feature_transfer import BASELINE_MANIFEST, ENCODER, STUDY


def diagnose(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(BASELINE_MANIFEST.read_text())
    for file, expected in manifest["data_audit"]["source_sha256"].items():
        assert sha(STUDY / file) == expected
    prompts = json.loads((STUDY / "prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    length, fingerprints = {}, {}
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, add_special_tokens=False, truncation=False, verbose=False)
        length[prompt] = len(tokens)
        fingerprints[prompt] = hashlib.sha256(np.asarray(tokens[:2048], dtype="<i4").tobytes()).hexdigest()
        if (i+1) % 2000 == 0:
            print(f"Tokenized {i+1:,}/{len(prompts):,} prompts", flush=True)
    rows = pd.read_parquet(STUDY / "observations.parquet", columns=["dataset", "item_id", "item_content"])
    summary = []
    for benchmark, part in [("all", rows), *rows.groupby("dataset")]:
        for unit, texts in (("unique_prompt", part.item_content.drop_duplicates()),
                            ("unique_item", part.drop_duplicates("item_id").item_content),
                            ("response", part.item_content)):
            sizes = texts.map(length).to_numpy()
            summary.append(dict(benchmark=benchmark, weighting=unit, count=len(sizes),
                tokens=int(sizes.sum()), retained_tokens=int(np.minimum(sizes, 2048).sum()),
                token_fraction_retained=float(np.minimum(sizes, 2048).sum()/max(1, sizes.sum())),
                fully_covered_fraction=float((sizes <= 2048).mean()),
                median_tokens=float(np.median(sizes)), p95_tokens=float(np.quantile(sizes, .95)), max_tokens=int(sizes.max())))
    pd.DataFrame(summary).to_csv(destination / "coverage.csv", index=False)
    prompt_table = pd.DataFrame(dict(prompt_index=range(len(prompts)),
                                     tokens=[length[p] for p in prompts],
                                     retained_token_sha256=[fingerprints[p] for p in prompts]))
    prompt_table.to_csv(destination / "prompt_token_inventory.csv", index=False)
    groups = prompt_table.groupby("retained_token_sha256").size()
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=manifest["data_audit"]["source_sha256"],
        script_sha256=sha(Path(__file__)), encoder_source=json.loads((ENCODER / "source.json").read_text()),
        tokenizer_files_sha256={p.name: sha(p) for p in ENCODER.iterdir() if p.is_file() and
                                ("token" in p.name or p.name in {"vocab.txt", "config.json"})},
        content_token_cap=2048, unique_prompts=len(prompts),
        groups_with_identical_capped_token_input=int((groups > 1).sum()),
        prompts_in_those_groups=int(groups[groups > 1].sum()),
        limitation="Tokenizer coverage, not embedding quality or measured accuracy loss. Response weighting repeats prompts; unique-prompt weighting does not.")
    (destination / "VERIFICATION.json").write_text(json.dumps(receipt, indent=2))
    print(pd.DataFrame(summary).query("weighting == 'unique_prompt'").to_string(index=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    diagnose(parser.parse_args().destination)
