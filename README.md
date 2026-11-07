This branch starts at Steven's `steven/approach-2` commit
`5823c1ffad7c1ce5912fb648f46924b832561926` and adds Approach 1's text chunking.
Steven's November 7 unseen-subject evaluation fix (`435ab92`) is also merged.

[Full-data results and graphs](FULL_DATA_RESULTS_2026-11-07.md) are the current
evaluation: item validation Brier 0.1920, independent item-test Brier 0.1977 and
macro benchmark-test Brier 0.3053 (equal-pair weighting; lower is better).
All 15 regression checks and the full saved-prediction audit passed.

Use `approach2_full_data_holdouts.ipynb` for the full-data experiment: **229,143
eligible responses**, 10,628 item IDs and 9,688 unique prompt vectors. This uses
the latest Approach 1 cohort and exact recorded item/benchmark splits. It verifies
raw source hashes, response identities, labels, subject names and frozen-vector
cache provenance before reuse. It does not sample responses or overwrite the
earlier 1,000-row pilot.

The full local download contains 571,921 response rows. The established Approach
1 correctness-prediction policy retains all rows from four binary benchmarks
(136,282) plus MathArena outcomes exactly equal to numeric 0 or 1 (92,861).
MMDocRAG's 342,695 graded-credit rows and 83 fractional MathArena responses remain
excluded; no scores are rounded or converted to binary labels. Retained MathArena
endpoints include some interval-scored criteria, so that population is conditional
on the endpoint filter.

The full-data notebook uses a separate outer item test and internal validation,
plus five complete benchmark holdouts. The item split contains 139,782 training,
47,491 validation and 41,870 outer-test responses. Unseen subject names are scored
using the existing mean-ability fallback. All runs and graphs are versioned under
`results/chunked/experiments/`; the corresponding executed notebook is saved under
`benchmark_results/`. The model architecture, chunk budget and training objective
remain unchanged. Training a held-out model does not train the all-data submission
model.

The same MiniLM encoder produces one 384-dimensional vector per prompt. It retains
up to 2,048 content tokens in nonoverlapping windows of at most 512 tokens including
special tokens (510 content tokens per window, up to five windows). Each chunk
vector is L2-normalized; pooling takes a content-token-weighted mean and normalizes
again. The IRT architecture and Brier/MSE loss are unchanged; the training notebook
below adds explicit holdouts and validation checkpoint selection.

Run commands from the repository root with the project's Python environment:

```bash
# Reuse Steven's exact saved responses, labels and prompts; only replace embeddings.
python reembed_samples.py
python models/Semantic_IRT.py
python -c 'import utils; utils.create_submission_zip("submission_chunked")'
```

Alternatively, run `preprocess.ipynb` for a newly sampled dataset. Both paths save
`samples_chunked.json`; neither overwrites it if it already exists. Training reads
that file by default and writes `results/chunked/`. Prediction and the default ZIP
use `results/chunked/all_data/`. The original `samples.json`, `results/all_data/`
and baseline fold results remain intact.

Re-embedding and retraining are required before prediction/export: Steven's saved
weights were fitted to truncated embeddings. Encoding metadata on each sample
and in the training summary prevents accidentally mixing the two pipelines.
The shared implementation is `models/text_chunking.py`, lifted from
`my-approach1/approach1_submission/model.py`; only its loader was adapted to accept
the existing Hugging Face model name. No new dependency is introduced.

Run the chunking checks with a local MiniLM directory (or omit the variable to use
the normal Hugging Face model/cache):

```bash
PSN_IRT_TEST_ENCODER=/path/to/text_encoder python -m unittest discover -s tests -v
```

Initial port verification on November 7, 2026: all eight checks passed using the existing local MiniLM
encoder, including boundary/cap handling, weighted pooling, batch consistency,
tokenizer fallback, baseline rejection, response preservation and an isolated
one-epoch training → checkpoint load → prediction → ZIP run on synthetic data.
The three tokenization/pooling functions also match Approach 1's source exactly
by Python AST comparison. The subsequent full 1,000-response experiment and its
additional regression checks are linked below.

For training and graphs, open `approach2_chunked_holdouts.ipynb`. It follows the
Approach 1 notebook sequence: frozen embeddings, editable training settings,
item/prompt-group splits, epoch-end learning curves, and constant-baseline
comparisons. It runs an 80/20 item holdout and all four leave-one-benchmark-out
experiments on Steven's exact 1,000 saved responses. Each benchmark experiment
uses an internal item/prompt-held-out validation set from the other benchmarks
for checkpoint selection. The entire outer benchmark is scored afterward.
Unseen subject names use the existing population-mean ability fallback and stay
in the reported denominator; known/unseen counts and known-only scores are saved.

The [November 7 results](HOLDOUT_RESULTS_2026-11-07.md) link the executed notebook,
all four graphs and the verification receipt. Item validation beats both constant
baselines; whole-benchmark transfer is mixed. All 13 regression checks passed.

The notebook writes versioned runs to `results/chunked/experiments/`, with model
checkpoints, response-level split/prediction CSVs, source hashes, settings and
four PNG graphs. Choose a new `RUN_NAME` to retrain; saved-results/plot cells can
be rerun without training. It does not replace the all-data submission model.

Steven's original trainer shuffled training-benchmark and held-out-benchmark
rows together. This branch now fixes that boundary with explicit subsets.
Its CLI still uses a response-row internal validation split; the merged upstream
fix now also scores unseen subjects using average training-subject ability.
Use the new notebook for the grouped evaluation described above. Old baseline fold scores remain
historical and are not comparable to the corrected benchmark experiment.

Steven's original workflow notes follow (paths above supersede the baseline paths).

Run preprocess.ipynb to generate lookup tables and item embeddings.

Run Semantic_IRT.py to locally train/validate/test. Generates tabular results, graphical results, and a "best" model for each LOBO fold, then one more trained on all available data (none held out, this is the one that gets packaged into submission zip).

Call function utils.create_submission_zip() to create submission zip to upload to CodaBench. You can also call tools/check_submission_zip.py and tools/run_smoke_test.py (sanity checks provided by the competition organizers) before uploading.
