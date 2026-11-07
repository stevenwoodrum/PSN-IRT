# Could a stronger embedding model help?

**Yes, this is a reasonable experiment; the evidence does not establish MiniLM as the main bottleneck.** Both previous approaches use the same frozen MiniLM vectors, yet Approach 1 M19 scored 0.179812 on outer items and the old Semantic IRT model scored 0.193620. The shared encoder therefore cannot by itself explain that gap. Architecture, metadata, optimization and calibration can affect how much either model extracts from those vectors.

The [new feature-transfer experiment](FEATURE_TRANSFER_RESULTS_2026-11-07.md) provides a more direct diagnostic within one ordered-4PL recipe. Full metadata/text scored 0.199111 on internal validation and 0.197681 on outer items. Removing text scored 0.203555 and 0.201783. Current embeddings contribute in that comparison, but the modest gain does not measure what another representation could add. Every model was refitted; this is not an invalid test that zeroes an already trained model's inputs. One seed, changed input dimension and separate early stopping limit causal precision.

## What the encoder is losing

The current [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes 384-dimensional sentence/paragraph embeddings aimed at semantic similarity and retrieval. Those objectives do not directly train a representation of model-specific task difficulty. Our implementation overrides the default short sequence limit and uses the explicitly verified 512-token chunks; it is **not silently restricted to 256 tokens**.

However, pooling separate windows cannot let tokens attend across windows. Averaging can dilute a short crucial constraint among background text. The total 2,048-content-token cap also removes all later material. A better encoder might improve the retained-text representation without addressing either loss.

`diagnose_prompt_coverage.py` tokenized all **9,688 distinct prompts** using the actual local tokenizer and verified cohort hashes. These are full-cohort figures, not the earlier 927-item pilot:

| Benchmark | Unique prompts | Fully within 2,048 content tokens | Aggregate tokens retained |
|---|---:|---:|---:|
| MathArena | 894 | 100.0% | 100.0% |
| Multi-SWE-bench | 2,078 | 96.8% | 94.0% |
| WebAgents | 233 | 100.0% | 100.0% |
| ResearchCodeBench | 212 | 2.8% | **5.8%** |
| SWE-rebench | 6,271 | 96.7% | 84.9% |
| Entire cohort, deduplicated prompts | 9,688 | 95.0% | 37.1% |

ResearchCodeBench's median prompt is 35,751 MiniLM content tokens, and its longest is 54,192. Even 32K context is not guaranteed to cover these prompts, especially with a different tokenizer. Their actual important locations need inspection before choosing head/tail selection or a structured summary. Token retention alone is not measured predictive information or accuracy loss.

The per-benchmark prompt counts are not additive: some texts occur in more than one benchmark. Response-weighted coverage retains 28.1% of aggregate tokens; unique-item weighting retains 38.0%. Those weightings deliberately differ from the 37.1% unique-prompt figure. The [coverage CSV](reports/feature_transfer_20261107/prompt_coverage.csv) is included in Git; the full token inventory and original provenance receipt remain local.

There are also 34 groups containing 241 distinct raw prompts with identical retained token sequences. Tokenizer normalization as well as truncation can produce this; it is not proof that all these collisions were caused by the cap.

## Other limitations a text encoder cannot repair

- In the historical model, all unseen names get the same mean ability. The new metadata model can use the other two fields, but cannot recover unavailable subject/configuration distinctions.
- Full A1 metadata still collapses 104 MathArena IDs into 53 tuples and 82 Multi-SWE-bench IDs into 20 tuples. Adding raw IDs as a shortcut was not part of A1 and was not done here.
- One scalar ability is a restrictive skill representation. Better text does not introduce a multidimensional or domain-conditioned subject profile by itself.
- The old unordered 4PL bounds and inconsistent parameter export were real mathematical problems. They are corrected in the new version, but the new scores show that correction alone does not ensure better predictions.
- Large held-out-domain calibration errors and the row-versus-equal-pair objective mismatch remain candidate explanations. More embedding dimensions do not guarantee calibrated probabilities.

## The next encoder comparison

I would test **Qwen3-Embedding-0.6B** as a modern candidate. Its publisher documents code-related embedding evaluations, 32K context, and output dimensions from 32 to 1,024. Those are reasons to test it, not evidence of better Brier scores on this competition. [Official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

**BGE-M3** is another candidate: its publisher specifies 1,024-dimensional dense vectors and up to 8,192 tokens. Use its actual saved pooling/tokenization, not the current BERT-specific helper. [Official model card](https://huggingface.co/BAAI/bge-m3).

Use a small, predeclared comparison rather than switching both encoder and IRT recipe and attributing any gain to embeddings:

1. Fix the cohort, recorded partitions, metadata policy, IRT recipe, training budget, primary metrics and seed list. Include the current MiniLM baseline. Choose recipes using internal validation only; reuse of the old outer sets remains exploratory.
2. For an encoder-pipeline comparison, give both encoders the **same literal original-text prefix**, with its boundary derived from MiniLM token offsets. Equal token counts under different tokenizers do not guarantee equal content. Use each model's supported native pooling and document that these encoding pipelines differ. Do not retrofit Qwen's decoder pooling into the current BERT `[CLS]/[SEP]` code.
3. Keep the downstream input dimension controlled where supported. Qwen's documented dimensional truncation permits a 384-dimensional comparison; a full-dimensional run is a distinct capacity experiment. For other encoders, a projection/PCA must be fitted on training prompts only, with the parameter-count difference disclosed.
4. Separately compare a larger text view with the **same new encoder**. This distinguishes changing the encoder pipeline from exposing previously discarded content. Predeclare budget/selection rules and record the actual text spans and tokenizer coverage, especially for ResearchCodeBench.
5. Recompute and version the prompt-vector cache, pin the encoder revision and hash model/tokenizer files, pooling, instructions, dimensions, text-selection policy and libraries. Retrain every head. Require batch/single consistency and train/inference parity. Keep both current caches and frozen run snapshots.
6. Report per-benchmark and known/unseen results, calibration, row/equal-pair Brier, validation selection, repeated seeds, encoding time and inference cost. A higher retrieval benchmark score is not a substitute for those measurements.

No alternative encoder was downloaded or evaluated here. The completed work is the feature/formula transfer, 25 fits, a refitted no-text control, and full-cohort coverage diagnosis. A stronger encoder remains a worthwhile **testable hypothesis**, particularly alongside better long-document handling.
