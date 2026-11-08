# Approach 2: chunked Semantic IRT on all eligible samples

This branch extends Steven's Semantic IRT with chunked text embeddings, the full
**229,143-response** cohort and grouped item/benchmark holdouts. It contains:

- an experimental model using Approach 1's subject features;
- the proposal's scalar-ability 2PL model, which uses all subject and item metadata
  ([details below](#proposed-2pl-model-scalar-ability)).

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

- **Proposed 2PL with all metadata:** see [Reproduce](#reproduce) below.

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

The feature-transfer variants in the table **have not beaten the original model**.
All benchmark averages above lose to constant probability 0.5 (Brier 0.25).
These are single-seed, reused test sets, not hosted competition results.

Tables, graphs and the verification summary for these runs are in
[reports/feature_transfer_20261107/](reports/feature_transfer_20261107/).

## Proposed 2PL model (scalar ability)

This is the proposal's Approach 2, keeping a **single ability number** per subject:

**P(correct) = sigmoid(a_i · (θ_s − b_i))**

- **Subject network:** gives the ability θ_s.
- **Item network:** gives the difficulty b_i and the discrimination a_i = softplus(·) + 0.1.
  Because a_i is always positive, the probability always rises with ability.

Compared with the original Semantic IRT:

| | Original | Proposed 2PL |
|---|---|---|
| Formula | 4PL: c + (d − c)·σ(a(θ − β)) | 2PL: σ(a(θ − b)); no c (guessing) or d (ceiling) |
| Loss | MSE | Binary cross-entropy |
| Subject input | one-hot `normalized_name` only | one-hot of all eight subject fields `predict()` receives (see below) |
| Item input | MiniLM chunked text vector (384) | the same vector + multi-hot `item_features` tokens |

The eight subject fields are `normalized_name`, `provider`, `reasoning_effort`, `release_date`,
`access_date`, `harness`, `harness_version` and `subject_features_extra`. Together they uniquely
identify every subject: MathArena 104/104 and Multi-SWE-bench 82/82, compared with 46 and 20
distinct names.

The `item_features` tokens are MathArena competition and problem number, Multi-SWE-bench language,
WebAgents website and ResearchCodeBench paper. SWE-rebench has none.

Categories and token vocabularies are fitted on training rows only. Missing values become an
empty string and unseen values become zeros. Subject and benchmark IDs and outcomes are never
used as features.

Everything else is unchanged:

- hidden size 64;
- AdamW with learning rate 3e-4 and weight decay 1e-5, batch 512;
- up to 100 epochs, early-stopping patience 10;
- the same frozen MiniLM vectors, cohort and recorded splits;
- checkpoints selected by internal-validation equal-pair Brier.

The legacy and ordered-4PL models and their checkpoints are untouched; this model is versioned
`proposed_2pl_scalar_v1`.

### Results

Equal subject/benchmark-pair Brier, lower is better. The proposed rows are the mean ± sd over
seeds 5814, 5815 and 5816. The baselines are seed 5814.

| Model | Item test | Benchmark-held-out average |
|---|---:|---:|
| Constant 0.5 | 0.250 | **0.250** |
| Original Semantic IRT (4PL) | 0.194 | 0.308 |
| Approach 1 M03 (FNN) | 0.186 | 0.303 |
| 2PL, Approach 1's 3 subject fields | 0.202 ± 0.010 | 0.338 ± 0.007 |
| + 5 extra subject fields | 0.186 ± 0.003 | 0.344 ± 0.025 |
| + `item_features` (no extra subject fields) | 0.164 ± 0.002 | 0.339 ± 0.004 |
| **2PL with all metadata (`2pl_full`)** | **0.152 ± 0.001** | 0.340 ± 0.025 |

In short:

- **Proposed 2PL vs Approach 1 M03:** item-test Brier was 0.152 vs 0.186, and the
  benchmark-held-out average was 0.340 vs 0.303.
- **Where the gain comes from:** the added metadata, mainly `item_features`, plus the full subject
  fields in combination. The 2PL/BCE change alone (first 2PL row) did not help.
- **Unseen benchmarks are not solved:** every model is still worse than constant 0.5 there.
- **Selection:** internal validation chose `2pl_full` for every seed before any outer test was scored.

Caveats:

- Outer test sets were examined in earlier work, so this is not a fresh test.
- Approach 1 never received these inputs, so this is not an architecture-only comparison.
- The 4PL has not been tested with the full inputs.
- The best epoch was 1–2 throughout.

Compact tables, the figure and audit summaries are in
[reports/proposed_2pl_20261107/](reports/proposed_2pl_20261107/).

### Reproduce

The data and Python requirements are those in [Setup](#setup) and [Data](#data). Each command
fits 4 recipes × 6 splits (24 fits; about 7 minutes per seed on this Mac's Apple GPU). Use fresh run
names; existing run folders are never overwritten.

```bash
for seed in 5814 5815 5816; do
  python run_proposed_irt.py --run-name proposed_2pl_seed$seed --seed $seed
done
```

Audit each run. The audit re-joins every feature from the raw parquet files, checks split
membership, reproduces the checkpoint predictions and recomputes the scores:

```bash
for seed in 5814 5815 5816; do
  python verify_proposed_irt.py results/chunked/experiments/proposed_2pl_seed$seed
done
```

Build the comparison tables:

```bash
python summarize_proposed_irt.py results/chunked/experiments/proposed_2pl_seed{5814,5815,5816} \
  --out reports/proposed_2pl_repro
```

The summary step also reads the saved baseline runs (`chunked_full_data_mps_20261107_v1`,
`feature_transfer_mps_20261107_v1`) and Approach 1's M03 predictions. Those are local artifacts,
not part of Git.

Exact scores can vary slightly by device. The recorded fits used Apple GPU (`--device mps`).

Code:

- `models/proposed_irt.py`: model and encoders
- `proposed_training.py`: training and evaluation
- `run_proposed_irt.py`: runner
- `verify_proposed_irt.py`: audit
- `summarize_proposed_irt.py`: tables
- `tests/test_proposed_irt.py`: tests

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
