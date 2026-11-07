# November 7 feature-transfer report

This is the compact, shareable output of 25 full-cohort training fits. The full
response CSVs, checkpoints and original immutable run receipts remain local.

- `comparison.csv`: row/equal-pair Brier and BCE for every evaluated partition.
- `benchmark_macro.csv`: equally weighted average of five outer benchmark scores.
- `selection.json`: recipe selection locked using internal validation before
  scoring outer item responses.
- `verification_summary.json`: summary of the successful independent audit of
  all 25 fits; exported parameters reproduced probabilities within 1.97e-7.
- `identity_audit.csv`: subject IDs versus names and metadata tuples.
- `prompt_coverage.csv`: full-cohort coverage under the 2,048-content-token cap.
- `data_provenance.json`: frozen cohort/vector hashes used by the loader.
- `library_versions.json`: the environment used for the recorded training.
- The three PNGs show item learning curves, item scores and benchmark scores.

The validation-selected dropout variant did not outperform the old baseline.
All results use one seed; the outer tests had been examined in earlier work.
The shared code now supports automatic device selection; those portability
changes do not change the recorded historical fits.
