"""Build the editable/viewable companion to the versioned feature suite."""

import json
from pathlib import Path


cells = []


def cell(kind, text):
    entry = dict(cell_type=kind, id=f"feature-transfer-{len(cells):02}", metadata={},
                 source=text.splitlines(keepends=True))
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell("markdown", """# Semantic IRT: Approach 1 features and controlled variants

Use the full 229,143-response binary/endpoint cohort and exact recorded item/prompt-group splits. Input fields are `normalized_name`, `provider`, `reasoning_effort`, and item text through frozen embeddings. All subjects stay in evaluation. No subject ID or benchmark ID is learned as an input feature.

This is a **new ordered 4PL model version**. Its probability is $c+(d-c)\\sigma(a(\\theta-\\beta))$, with $a=\\mathrm{softplus}(z_a)+0.1$, $c=0.3\\sigma(z_c)$, and $d=c+(1-c)\\sigma(z_d)$. Exports use these same parameters. Legacy weights retain their original interpretation and are not accepted by the new loader.

Training automatically selects an available accelerator or CPU. Set `DEVICE` to override this. Leave `RUN_TRAINING=False` to inspect the included compact report without training. Use a fresh run name before enabling training. Existing run directories are protected.
""")
cell("code", """import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
from feature_transfer import DEFAULT_SETTINGS, RECIPES
from run_feature_transfer import run_suite
WORKSPACE = Path.cwd()
assert (WORKSPACE / 'feature_transfer.py').is_file(), 'Launch from PSN-IRT.'
""")
cell("markdown", """## Training settings

All feature ablations keep the same partitions and training settings. Each fits categories on its training rows only; an unseen category produces zeros for that field. Missing metadata uses the empty string, as in Approach 1. The full metadata variant uses available provider/reasoning information even if a name is unseen. The corrected name-only control retains the old mean-ability fallback.

Ten item trials isolate the corrected name model, full metadata, each omitted metadata field, omitted text, dropout 0.2, explicit kernel L2 0.0001, hidden size 32, and BCE loss. AdamW weight decay remains 0.00001; explicit kernel L2 is a separate penalty, not a renamed weight decay. A new metadata representation still merges indistinguishable subject configurations.
""")
cell("code", """RUN_NAME = 'feature_transfer_v2'
RUN_DIR = WORKSPACE / 'results/chunked/experiments' / RUN_NAME
REPORT_DIR = WORKSPACE / 'reports/feature_transfer_20261107'
if not (RUN_DIR / 'complete.json').is_file():
    RUN_DIR = REPORT_DIR
SETTINGS = dict(DEFAULT_SETTINGS)
RECIPE_NAMES = [recipe.name for recipe in RECIPES]
DEVICE = 'auto'
RUN_TRAINING = False
display(pd.DataFrame([dict(name=r.name, fields=', '.join(r.fields), text=r.text,
                           hidden=r.hidden, dropout=r.dropout, kernel_l2=r.kernel_l2,
                           objective=r.objective) for r in RECIPES]))
print(SETTINGS)
""")
cell("markdown", """## Fit, choose with validation, then score

The runner fits all item trials without evaluating outer item responses, then writes `selection.json` choosing the recipe with the lowest internal-validation equal-pair Brier. It scores the outer item rows afterward. Ordered name-only, full metadata, and the validation-selected recipe then run all five benchmark holdouts, using internal validation for every checkpoint. A benchmark test never enters fitting or checkpoint selection.

These fixed outer sets have been inspected in earlier experiments: they are repeated evaluation, **not a fresh prospective test**. One seed cannot establish consistent superiority. This notebook does not create an all-data submission model or upload anything.
""")
cell("code", """if RUN_TRAINING:
    RUN_DIR = run_suite(RUN_NAME, SETTINGS, RECIPE_NAMES, device=DEVICE)
else:
    assert (RUN_DIR / 'complete.json').is_file(), 'Saved run unavailable or incomplete.'
display(json.loads((RUN_DIR / 'selection.json').read_text()))
display(pd.read_csv(RUN_DIR / 'identity_audit.csv'))
""")
cell("code", """comparison = pd.read_csv(RUN_DIR / 'comparison.csv')
learned = comparison[comparison.predictor == comparison.recipe]
item = learned[learned.experiment == 'item_holdout']
display(item.pivot(index='recipe', columns='partition', values='equal_pair_Brier'))
display(item[['recipe', 'partition', 'row_Brier', 'equal_pair_Brier', 'best_epoch']])
display(pd.read_csv(RUN_DIR / 'benchmark_macro.csv'))
outer = learned[(learned.partition == 'heldout') & (learned.experiment != 'item_holdout')]
display(outer.pivot(index='experiment', columns='recipe', values='equal_pair_Brier'))
""")
cell("markdown", """## Learning curves and comparisons

Row Brier weights responses equally. Equal-pair Brier first averages within each `(benchmark_id, subject_id)` and then weights those pairs equally. Benchmark macro weights the five benchmark test scores equally. Constant 0.5 has Brier 0.25. Keep these weightings distinct.
""")
cell("code", """for name in ['item_learning_curves.png', 'item_comparison.png', 'benchmark_comparison.png']:
    display(Image(filename=str(RUN_DIR / name)))
""")
cell("markdown", """## Verification and interpretation

`verify_feature_transfer.py RUN_DIR` independently checks source/cohort hashes, all recorded split memberships, training-only categories, checkpoint predictions, all subject rows, both Brier weightings, checkpoint selection, ordered parameter bounds, and probabilities reconstructed from exported numeric parameters. It automatically selects a device; `--device` overrides it. The compact report includes the original receipt; rerunning a full audit requires the local checkpoints and data. Inspect the receipt before treating a new run as verified.

A no-text ablation measures what this recipe gains from the current text pipeline; it does **not** measure what a stronger encoder could gain. An encoder upgrade needs a matched experiment with preserved text budgets, cache/model provenance, dimensions, and validation-only selection. Longer-context experiments should be distinguished from changes in encoder quality.
""")
cell("code", """receipt_path = RUN_DIR / 'VERIFICATION.json'
if not receipt_path.is_file():
    receipt_path = RUN_DIR / 'verification_summary.json'
if receipt_path.is_file():
    receipt = json.loads(receipt_path.read_text())
    display({key: value for key, value in receipt.items() if key not in ['folds', 'artifact_sha256']})
else:
    print('Independent audit has not yet completed.')
""")

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                       language_info=dict(name="python", version="3.12")), nbformat=4, nbformat_minor=5)
Path(__file__).with_name("approach2_feature_transfer.ipynb").write_text(json.dumps(notebook, indent=1)+"\n")
