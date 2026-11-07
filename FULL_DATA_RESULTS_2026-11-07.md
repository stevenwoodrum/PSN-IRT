Full-data training completed with **229,143 eligible response rows**, replacing
the 1,000-row pilot for current evaluation. This is 228,143 additional responses
and 229.143 times the pilot size. There are 10,628 distinct item IDs, 9,688 prompt
strings and 9,687 connected item/prompt groups.

The full local download contains 571,921 response rows:

| Benchmark | Downloaded rows | Included in training/evaluation cohort |
|---|---:|---:|
| MathArena | 92,944 | 92,861 |
| MMDocRAG | 342,695 | 0 |
| Multi-SWE-bench | 57,975 | 57,975 |
| WebAgents | 4,449 | 4,449 |
| ResearchCodeBench | 6,784 | 6,784 |
| SWE-rebench | 67,074 | 67,074 |
| **Total** | **571,921** | **229,143** |

This preserves the latest Approach 1 eligibility policy: four binary benchmarks
plus MathArena response values exactly equal to numeric 0 or 1. No values are
rounded. MMDocRAG is a graded-credit task, and MathArena has 83 excluded
fractional responses. Some retained MathArena endpoints come from interval-scored
proof criteria, so the evaluated population is conditional on that endpoint
filter; those criteria are not intrinsically binary.

[Executed full-data notebook](benchmark_results/chunked_full_data_20261107_v1/approach2_full_data_holdouts.ipynb)
and [editable notebook](approach2_full_data_holdouts.ipynb).

The original 4PL architecture, row-weighted MSE/Brier objective, 2,048-token
content cap, 512-token chunk windows and normalized token-weighted pooling remain
unchanged. The full verified Approach 1 prompt-vector cache was reused after
checking raw source hashes, row identities/labels/names and current
encoder/library/code cache keys. No Approach 1 artifacts were modified.

Seed 5814; AdamW learning rate 0.0003, weight decay 0.00001, batch size 512;
maximum 100 epochs; early stopping after 10 epochs without internal-validation
equal-pair Brier improvement. Training ran on CPU. The recorded Approach 1
membership and split indices are reused exactly. The item experiment has
139,782 training rows, 47,491 internal-validation rows and a separate outer item
test of 41,870 rows. Each outer benchmark is also entirely absent from its fold's
training and internal validation.

Scores below are **equal subject-benchmark-pair Brier; lower is better**.
Constant 0.5 scores 0.2500 in every partition.

| Evaluation | Scored rows | Model | Training-mean baseline | Selected epoch |
|---|---:|---:|---:|---:|
| Item internal validation | 47,491 | 0.1920 | 0.2430 | 1 |
| Independent item test | 41,870 | 0.1977 | 0.2422 | 1 |
| Held-out MathArena | 92,861 | 0.3991 | 0.3163 | 1 |
| Held-out Multi-SWE-bench | 57,975 | 0.2484 | 0.2968 | 8 |
| Held-out WebAgents | 4,449 | 0.2738 | 0.2424 | 1 |
| Held-out ResearchCodeBench | 6,784 | 0.2969 | 0.2426 | 2 |
| Held-out SWE-rebench | 67,074 | 0.3081 | 0.2523 | 2 |

The independent item test beats both constant baselines. Its response-weighted
Brier is 0.1934, versus 0.2444 for the training-mean baseline. Whole-benchmark
generalization remains weak: macro benchmark equal-pair Brier is 0.3053, versus
0.2500 for constant 0.5. Only Multi-SWE-bench narrowly beats constant 0.5 under
equal-pair weighting. Larger training data has not established reliable transfer
to new benchmarks. The pilot differs in cohort and split and is not a controlled
estimate of the effect of adding data.

Steven's [unseen-subject fix, commit 435ab92](https://github.com/stevenwoodrum/PSN-IRT/commit/435ab92c1645d449bc714757a43900d299062d79),
has been fetched and merged. It preserves unseen test rows and uses average
training-subject ability. Our notebook already scored all such rows before the
merge. It now calls his shared model fallback directly, while retaining our
separate correction to the legacy CLI's benchmark partition boundary.

Unseen-subject rows included in the outer scores: MathArena 75,611/92,861;
Multi-SWE-bench 13,460/57,975; WebAgents 1,660/4,449; ResearchCodeBench 3,392/6,784;
SWE-rebench 67,074/67,074. The independent item test has no unseen subject names.
**No outer test responses were dropped for having an unseen subject name.**

An independent audit verified every 229,143-row partition against the recorded
Approach 1 split, checked baseline means and recomputed row/equal-pair Brier.
Every checkpoint was loaded into the merged upstream model and all saved
predictions were reproduced within a maximum absolute difference of 1.3e-7.
The exact pre-merge training source remains in the run's `source/` snapshot;
compatibility with the merged model is recorded in the
[verification receipt](results/chunked/experiments/chunked_full_data_20261107_v1/VERIFICATION.json).
All 15 regression checks passed, including a test where unseen subjects must
remain in the legacy CLI's held-out denominator.

- [Training and internal-validation curves](results/chunked/experiments/chunked_full_data_20261107_v1/learning_curves.png)
- [Internal-validation comparison](results/chunked/experiments/chunked_full_data_20261107_v1/validation_comparison.png)
- [Independent item and benchmark scores](results/chunked/experiments/chunked_full_data_20261107_v1/heldout_scores.png)
- [Independent item test by benchmark](results/chunked/experiments/chunked_full_data_20261107_v1/item_test_by_benchmark.png)

This is a single-seed full-data run. No matched unchunked ablation, hyperparameter
search, repeated-seed uncertainty estimate, all-data submission training, hosted
upload or remote push was performed. The original pilot and all baseline
artifacts remain preserved.
