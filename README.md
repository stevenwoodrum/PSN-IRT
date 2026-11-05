Run preprocess.ipynb to generate lookup tables and item embeddings; saved alongside the originals in samples.json.

Run Semantic_IRT.py to locally train/validate/test. Generates tabular results, graphical results, and a "best" model for each LOBO fold, then one more trained on all available data (none held out, this is the one that gets packaged into submission zip).

Call function utils.create_submission_zip() to create submission zip to upload to CodaBench. You can also call tools/check_submission_zip.py and tools/run_smoke_test.py (sanity checks provided by the competition organizers) before uploading.