All six full-data experiments were rerun on the Mac's Apple GPU using PyTorch
MPS. Model parameters and every training/validation tensor were verified on
`mps:0`. Unsupported-operation CPU fallback was disabled. Each fold records its
device and positive MPS allocation in `settings.json`; allocations at the start
of training ranged from about 245 MB to 419 MB. Cached frozen embeddings were
reused; the text encoder did not need to run again.

[Executed GPU notebook and graphs](benchmark_results/chunked_full_data_mps_20261107_v1/approach2_full_data_holdouts.ipynb)
and [editable notebook](approach2_full_data_holdouts.ipynb).
The earlier CPU full-data run and 1,000-response pilot remain unchanged.

The comparison uses the same 229,143-row eligible dataset, chunked prompt
vectors, recorded item/benchmark splits and seed 5814. Scores are **equal
subject-benchmark-pair Brier; lower is better**. The benchmark column averages
the five independent outer benchmark scores equally. Internal validation selects
checkpoints; outer item/benchmark tests are never used for selection.

| Recipe | Item internal validation | Independent item test | Benchmark-test average |
|---|---:|---:|---:|
| Constant 0.5 | 0.2500 | 0.2500 | 0.2500 |
| Approach 1 original FNN, M03 | 0.1801 | 0.1858 | 0.3033 |
| Approach 1 dropout + L2, M08 | 0.1767 | 0.1815 | 0.3273 |
| Approach 1 64 units + dropout/L2, M09 | 0.1797 | 0.1854 | 0.3242 |
| Approach 1 Brier-loss variant, M19 | 0.1766 | 0.1798 | 0.3040 |
| **Approach 2 Semantic IRT, Apple GPU** | **0.1959** | **0.1936** | **0.3075** |

This run does **not establish an overall improvement over Approach 1**. Approach
2 is worse on held-out items than all four listed Approach 1 recipes. Its
benchmark average is slightly worse than the original FNN and the Brier-loss
variant, and better than the dropout/L2 variants. All listed learned models still
lose to constant 0.5 on the average held-out benchmark.

These are comparisons of complete training recipes: architecture, subject
features/fallback, loss, optimizer settings and batch size differ. This is not
an architecture-only or chunking-only ablation. Both approaches already use the
same chunked embeddings. Approach 1's item results here use seed 5814, not its
three-seed averages or hosted Codabench scores. Each selected Approach 1 source
table value was checked against its individual `complete.json` run receipt.

GPU outer scores:

| Outer test | Responses | Equal-pair Brier | Selected epoch |
|---|---:|---:|---:|
| Independent items | 41,870 | 0.1936 | 3 |
| MathArena | 92,861 | 0.3990 | 1 |
| Multi-SWE-bench | 57,975 | 0.2598 | 24 |
| WebAgents | 4,449 | 0.2741 | 1 |
| ResearchCodeBench | 6,784 | 0.2923 | 1 |
| SWE-rebench | 67,074 | 0.3124 | 2 |

All unseen-subject test responses remain included using Steven's merged
mean-ability fallback (`435ab92`). The independent item test has no unseen names.
The previous CPU full-data item-test score was 0.1977, with benchmark average
0.3053. CPU and MPS can use different random shuffling/numerical paths despite
the same seed, leading to different selected epochs. Their small score changes
are not evidence that GPU hardware improves predictive accuracy.

All 15 regression checks passed. A separate audit reloaded every checkpoint on
MPS, verified all device records and exact recorded split membership, reproduced
every saved response probability within maximum absolute difference 3e-8, and
independently recalculated row/equal-pair Brier and training-only baseline means.
The full training-source snapshot, prediction/split CSVs and manifest remain in
the versioned run directory.

- [Comparison CSV](results/chunked/experiments/chunked_full_data_mps_20261107_v1/matched_approach1_comparison.csv)
- [GPU verification receipt](results/chunked/experiments/chunked_full_data_mps_20261107_v1/VERIFICATION.json)
- [Training/internal-validation curves](results/chunked/experiments/chunked_full_data_mps_20261107_v1/learning_curves.png)
- [Outer item/benchmark scores](results/chunked/experiments/chunked_full_data_mps_20261107_v1/heldout_scores.png)

This remains one Approach 2 seed, with no repeated-seed uncertainty estimate or
matched unchunked ablation. No production all-data submission model, hosted
upload or remote push was performed.
