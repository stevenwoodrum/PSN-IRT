import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

import model
from models import text_chunking as chunking
import reembed_samples
import utils


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.encoder = chunking.make_text_encoder(
            os.environ.get("PSN_IRT_TEST_ENCODER", chunking.TEXT_ENCODING["model"])
        )

    def test_token_boundaries_and_cap(self):
        for count, expected in ((0, [1]), (510, [510]), (511, [510, 1]),
                                (2048, [510, 510, 510, 510, 8]),
                                (2200, [510, 510, 510, 510, 8])):
            with self.subTest(count=count):
                chunks = list(chunking.prompt_chunks(self.encoder, "cat " * count))
                self.assertEqual([weight for _, weight in chunks], expected)
                ids = [token for features, _ in chunks for token in features["input_ids"][1:-1]]
                self.assertEqual(ids, self.encoder.tokenizer.encode(
                    "cat " * count, add_special_tokens=False
                )[:2048])
                self.assertTrue(all(len(features["input_ids"]) <= 512 for features, _ in chunks))

    def test_pooling_uses_content_lengths(self):
        vectors = self.encoder.encode(["cat " * 510, "dog"], normalize_embeddings=True)
        expected = 510 * vectors[0] + vectors[1]
        expected /= np.linalg.norm(expected)
        actual = chunking.embed_text(self.encoder, ["cat " * 510 + "dog"])[0]
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        unweighted = vectors.sum(axis=0)
        unweighted /= np.linalg.norm(unweighted)
        self.assertGreater(np.linalg.norm(actual - unweighted), 0.01)

    def test_batching_empty_and_single_chunk(self):
        prompts = ["", "A small task", "cat " * 511]
        single = chunking.embed_text(self.encoder, prompts, batch_size=1)
        batched = chunking.embed_text(self.encoder, prompts, batch_size=3)
        np.testing.assert_allclose(single, batched, atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(batched[:2], self.encoder.encode(
            prompts[:2], normalize_embeddings=True
        ), atol=2e-6)
        self.assertEqual(chunking.embed_text(self.encoder, []).shape, (0, 384))
        with self.assertRaises(ValueError):
            chunking.embed_text(self.encoder, prompts, batch_size=0)

    def test_legacy_tokenizer_fallback(self):
        tokenizer = self.encoder.tokenizer
        fallback = SimpleNamespace(cls_token_id=tokenizer.cls_token_id,
                                   sep_token_id=tokenizer.sep_token_id, encode=tokenizer.encode)
        for ids in ([], [123, 456]):
            self.assertEqual(dict(chunking.prepare_chunk_features(tokenizer, ids)),
                             chunking.prepare_chunk_features(fallback, ids))

    def test_baseline_artifacts_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_summary.json").write_text('{}')
            model._get_predictor.cache_clear()
            try:
                with patch.object(model, "_ALL_DATA_DIR", root):
                    with self.assertRaisesRegex(ValueError, "Retrain"):
                        model._get_predictor()
            finally:
                model._get_predictor.cache_clear()

    def test_predict_uses_the_same_chunked_vector(self):
        from models.Semantic_IRT import Semantic_IRT

        network = Semantic_IRT(num_students=1, item_embed_dim=384).eval()
        predictor = model._Predictor(network, {"known": 0}, 0.0, 384, torch.device("cpu"))
        prompt = "cat " * 510 + "dog"
        embedding = torch.from_numpy(chunking.embed_text(self.encoder, [prompt]))
        with torch.inference_mode():
            expected = network(torch.ones(1, 1), embedding).item()
        with patch.object(model, "_get_predictor", return_value=predictor), \
                patch.object(utils, "get_sbert", return_value=self.encoder):
            actual = model.predict([{"normalized_name": "known"}, {"item_content": prompt}], [])
            unknown = model.predict([{"normalized_name": "new"}, {"item_content": prompt}], [])
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertTrue(np.isfinite(unknown) and 0 <= unknown <= 1)

    def test_reembedding_preserves_responses_and_baseline(self):
        baseline = [{"item_id": "item", "content": "task", "response": value,
                     "content_embedding": [99]} for value in (0, 1)]
        previous = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                source = Path("samples.json")
                source.write_text(json.dumps(baseline))
                original = source.read_bytes()
                with patch.object(utils, "get_sbert", return_value=self.encoder):
                    reembed_samples.main()
                updated = json.loads(Path("samples_chunked.json").read_text())
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual([s["response"] for s in updated], [0, 1])
                self.assertEqual(updated[0]["content_embedding"], updated[1]["content_embedding"])
                self.assertEqual(updated[0]["text_encoding"], chunking.TEXT_ENCODING)
                with self.assertRaises(FileExistsError):
                    reembed_samples.main()
            finally:
                os.chdir(previous)

    def test_training_prediction_and_zip_in_isolation(self):
        from models.Semantic_IRT import main as train

        previous = Path.cwd()
        source_root = Path(model.__file__).parent
        vector = chunking.embed_text(self.encoder, ["a small task"])[0].tolist()
        samples = [
            {"benchmark_id": benchmark, "normalized_name": subject,
             "item_id": f"{benchmark}-{item}", "response": item % 2,
             "content_embedding": vector, "text_encoding": chunking.TEXT_ENCODING}
            for benchmark in ("a", "b", "c") for item in range(6)
            for subject in ("known", "other")
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(root)
                Path("samples_chunked.json").write_text(json.dumps(samples))
                with patch("sys.argv", ["Semantic_IRT.py", "--epochs", "1"]):
                    train()
                artifacts = root / "results/chunked/all_data"
                summary = json.loads((artifacts / "training_summary.json").read_text())
                self.assertEqual(summary["text_encoding"], chunking.TEXT_ENCODING)
                self.assertFalse((root / "results/all_data").exists())
                model._get_predictor.cache_clear()
                with patch.object(model, "_ALL_DATA_DIR", artifacts), \
                        patch.object(utils, "get_sbert", return_value=self.encoder):
                    probability = model.predict(
                        [{"normalized_name": "known"}, {"item_content": "a small task"}], []
                    )
                self.assertTrue(np.isfinite(probability) and 0 <= probability <= 1)
                for name in ("model.py", "utils.py", "models.txt", "requirements.txt",
                             "models/Semantic_IRT.py", "models/text_chunking.py"):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_root / name, target)
                utils.create_submission_zip("submission_chunked")
                with ZipFile("submission_chunked.zip") as archive:
                    self.assertIn("models/text_chunking.py", archive.namelist())
                    self.assertIn("results/chunked/all_data/training_summary.json", archive.namelist())
                    self.assertNotIn("samples_chunked.json", archive.namelist())
            finally:
                model._get_predictor.cache_clear()
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
