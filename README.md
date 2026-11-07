# Approach 2: chunked Semantic IRT on all eligible samples

This branch extends Steven's Semantic IRT with chunked text embeddings, the full
**229,143-response** cohort, grouped item/benchmark holdouts, and an experimental
model using Approach 1's subject features.

## Setup

Use Python 3.12 and run commands from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training automatically selects CUDA, Apple GPU, or CPU. Use `--device cpu` or
`--device cuda` to override. Results can vary across devices and seeds.

## Data

Full-cohort training needs the shared Approach 1 study/data/cache folders; they
are **not included in this Git repository**. The default layout is:

```text
parent/
  PSN-IRT/                         # this repository
  approach1-starter/
    data/raw/                      # original benchmark parquet tables
    resources/text_encoder/        # saved MiniLM model/tokenizer + source.json
  my-approach1/
    prompt_embedding_cache.py
    experiments/
      approach1_comparison_20261020/
      approach1_matharena_binaryonly_20261020/
```

Keep the study's observations, prompts, vectors, source snapshots, embedding
caches, audits and recorded splits together. For another location, set
`PSN_IRT_STUDY`, `PSN_IRT_RAW_DATA`, and `PSN_IRT_ENCODER`. Match the recorded
[library versions](reports/feature_transfer_20261107/library_versions.json) when
reusing the frozen cache: its hashes and provenance are checked before training.
The tracked `samples.json` is Steven's earlier 1,000-row pilot, not the full data.

The cohort includes four binary benchmarks plus exact 0/1 MathArena outcomes.
MMDocRAG and 83 fractional MathArena responses are excluded without rounding.

## Run

- **Original model, full cohort:** open `approach2_full_data_holdouts.ipynb`.
- **Feature-transfer experiment:** open `approach2_feature_transfer.ipynb`, or run:

```bash
python run_feature_transfer.py --run-name feature_transfer_v2
python verify_feature_transfer.py results/chunked/experiments/feature_transfer_v2
```

Choose a fresh run name; existing runs are protected. Both workflows use the
same recorded item/prompt-group splits. Checkpoints are selected using internal
validation. Outer tests retain every subject, including unseen names. The feature
notebook can display the included report without the full training data.

## Current results

Equal subject/benchmark-pair Brier, lower is better:

| Recipe | Item test | Five-benchmark average |
|---|---:|---:|
| Original chunked Semantic IRT | 0.1936 | 0.3075 |
| Ordered 4PL + full subject metadata | 0.1977 | 0.3328 |
| Full metadata + dropout, validation-selected | 0.1971 | 0.3159 |

The new variants **have not beaten the original model**. All benchmark averages
above lose to constant probability 0.5 (Brier 0.25). These are single-seed,
reused test sets, not hosted competition results.

[Experiment details](FEATURE_TRANSFER_RESULTS_2026-11-07.md) ·
[Tables, graphs and verification summary](reports/feature_transfer_20261107/) ·
[Embedding/long-document assessment](EMBEDDING_ASSESSMENT_2026-11-07.md).

## Tests and submission

```bash
python -m unittest discover -s tests -v
```

Tests load MiniLM from the model cache, or set `PSN_IRT_TEST_ENCODER` to a local
model directory. Large checkpoints, response predictions and full data remain
local. The ordered-4PL model is separately versioned; it does not reinterpret
old weights. Holdout checkpoints are not production submission bundles.

The legacy sample-based submission path remains `reembed_samples.py` →
`models/Semantic_IRT.py` → `utils.create_submission_zip()`. It is separate from
the full-cohort experiments; no new all-data submission bundle has been trained.
