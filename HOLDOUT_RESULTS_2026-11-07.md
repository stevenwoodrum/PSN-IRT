First training run completed on Steven's exact saved 1,000 responses (927 item
IDs, 924 distinct prompt groups), seed 5814. The encoder uses the agreed 2,048
content-token cap and 512-token windows. The original 4PL architecture and
row-weighted MSE/Brier objective are retained. AdamW uses learning rate 0.0003,
weight decay 0.00001 and batches of 512, with at most 100 epochs, internal
validation checkpoint selection and patience 10.

[Executed notebook with inline graphs](benchmark_results/chunked_holdouts_20261107_v1/approach2_chunked_holdouts.ipynb)
and [editable notebook](approach2_chunked_holdouts.ipynb).
The notebook can replot saved CSVs without retraining; choose a new run name
before training again.

Scores below are **equal subject-benchmark-pair Brier; lower is better**.
The constant-0.5 baseline scores 0.2500 in every partition.

| Evaluation | Scored rows | Model Brier | Training-mean baseline | Selected epoch |
|---|---:|---:|---:|---:|
| Item-held-out validation | 198 | 0.1457 | 0.1723 | 19 |
| Held-out Multi-SWE-bench | 436 | 0.1589 | 0.2530 | 1 |
| Held-out WebAgents | 42 | 0.2919 | 0.2759 | 21 |
| Held-out ResearchCodeBench | 49 | 0.2343 | 0.2289 | 15 |
| Held-out SWE-rebench | 473 | 0.3770 | 0.3687 | 84 |

Item validation uses 802 training rows and 198 validation rows, grouped by item
ID and identical nonempty prompt. Its response-weighted Brier is 0.1821 versus
0.2188 for the training-mean baseline. It is checkpoint-selection validation,
not an untouched final test.

Every outer benchmark is absent from both training and checkpoint selection.
Each fold instead reserves 20% of item/prompt groups from the other benchmarks
as internal validation. No cross-boundary duplicate prompt required purging in
this sample. The old shuffled benchmark split was also corrected in the legacy
CLI, whose internal validation and unseen-subject filtering otherwise remain
unchanged. Its old fold scores are not comparable to this experiment.

Unseen subjects remain included through the existing mean-ability fallback:
102/436 Multi-SWE-bench, 16/42 WebAgents, 29/49 ResearchCodeBench and all 473/473
SWE-rebench rows. These counts refer to names actually present in each fold's
training split, not names merely present somewhere outside the held-out benchmark.
Five item-validation rows also need this fallback.

The item holdout beats both constant baselines. Benchmark generalization is
mixed: Multi-SWE-bench beats both; ResearchCodeBench beats constant 0.5 but
trails the training mean; WebAgents and SWE-rebench trail both. Macro benchmark
equal-pair Brier is 0.2655, compared with 0.2500 for constant 0.5. This single-seed
sample is not evidence of a consistent benchmark-transfer improvement, and no
matched unchunked run has been performed to isolate the effect of chunking.

All four plots are saved under the run directory:

- [Training and internal-validation curves](results/chunked/experiments/chunked_holdouts_20261107_v1/learning_curves.png)
- [Internal-validation baseline comparison](results/chunked/experiments/chunked_holdouts_20261107_v1/validation_comparison.png)
- [Outer benchmark scores](results/chunked/experiments/chunked_holdouts_20261107_v1/benchmark_holdout_scores.png)
- [Item validation by benchmark](results/chunked/experiments/chunked_holdouts_20261107_v1/item_validation_by_benchmark.png)

All 13 regression checks passed. An independent audit accounted for all 1,000
rows in each of five experiments, checked disjoint item/prompt groups and
benchmark membership, reproduced predictions from every saved checkpoint,
recomputed row and equal-pair scores, and checked training-only baseline means.
Its [verification receipt](results/chunked/experiments/chunked_holdouts_20261107_v1/VERIFICATION.json)
accompanies the split/prediction CSVs, checkpoints and source hashes. The exact
training source is preserved under the run's `source/` directory. Graph labels
were refreshed after training for readability without changing scores; the
executed notebook records the plotting-code hash.

No repeated-seed uncertainty estimate, larger-dataset experiment, all-data
submission training, hosted submission or remote push was performed.
