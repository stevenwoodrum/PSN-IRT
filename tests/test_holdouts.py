import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import holdout_training as training
import model


class HoldoutTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame([
            {"row_id": i, "item_id": f"item{i}", "content": f"prompt {i}",
             "benchmark_id": "a" if i < 6 else "b", "subject_id": "subject",
             "normalized_name": "known", "response_id": str(i), "response": i % 2}
            for i in range(12)
        ])
        frame["item_group"] = training.item_groups(frame)
        return frame

    def test_item_holdout_keeps_ids_and_duplicate_prompts_together(self):
        frame = self.frame()
        frame.loc[1, "content"] = frame.loc[0, "content"]
        frame.loc[2, "item_id"] = frame.loc[1, "item_id"]
        frame["item_group"] = training.item_groups(frame)
        self.assertEqual(frame.item_group.iloc[:3].nunique(), 1)
        split = training.make_split(frame, seed=5814)
        self.assertEqual(len(split["train"]) + len(split["validation"]), len(frame))
        self.assertTrue(set(split["train"].item_group).isdisjoint(split["validation"].item_group))

    def test_benchmark_rows_and_duplicate_prompts_never_enter_training_or_validation(self):
        frame = self.frame()
        frame.loc[0, "content"] = frame.loc[6, "content"]
        frame["item_group"] = training.item_groups(frame)
        split = training.make_split(frame, seed=5814, benchmark="b")
        self.assertEqual(set(split["heldout"].benchmark_id), {"b"})
        self.assertEqual(set(split["train"].benchmark_id), {"a"})
        self.assertEqual(set(split["validation"].benchmark_id), {"a"})
        self.assertEqual(split["purged"].row_id.tolist(), [0])

    def test_row_and_pair_weighting_are_distinct(self):
        frame = pd.DataFrame({"benchmark_id": ["a"] * 3, "subject_id": ["x", "x", "y"], "response": [0, 0, 1]})
        actual = training.score(frame, [0, 0, 0])
        self.assertAlmostEqual(actual["row_Brier"], 1 / 3)
        self.assertAlmostEqual(actual["equal_pair_Brier"], 0.5)

    def test_unseen_subject_fallback_matches_prediction_entrypoint(self):
        network = training.Semantic_IRT(2, item_embed_dim=384).eval()
        frame = self.frame().iloc[:2].copy()
        frame.loc[frame.index[1], "normalized_name"] = "unseen"
        vectors = {item: np.ones(384, dtype=np.float32) / np.sqrt(384) for item in frame.item_id}
        subjects = {"known": 0, "other": 1}
        data = training.tensors(frame, vectors, subjects, "cpu")
        actual = training.predict_tensors(network, data)
        with torch.inference_mode():
            mean = network.student_net(torch.eye(2)).mean().item()
        predictor = model._Predictor(network, subjects, mean, 384, torch.device("cpu"))
        with patch.object(model, "_get_predictor", return_value=predictor), \
                patch.object(model.utils, "get_sbert", return_value=None), \
                patch.object(model.utils, "embed_text", return_value=np.ones((1, 384), dtype=np.float32) / np.sqrt(384)):
            expected = [model.predict([{"normalized_name": name}, {"item_content": "task"}], [])
                        for name in frame.normalized_name]
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_training_saves_selected_checkpoint_and_split_receipt(self):
        frame = self.frame()
        split = training.make_split(frame, seed=5814, benchmark="b")
        vectors = {item: np.ones(384, dtype=np.float32) / np.sqrt(384) for item in frame.item_id}
        settings = {"seed": 5814, "epochs": 2, "batch_size": 4, "patience": 2,
                    "learning_rate": 3e-4, "weight_decay": 1e-5}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            results = training.run_fold(split, vectors, directory, settings, "test")
            history = pd.read_csv(directory / "training_history.csv")
            selected = int(history.loc[history.validation_equal_pair_Brier.idxmin(), "epoch"])
            self.assertTrue((results.best_epoch == selected).all())
            saved = pd.read_csv(directory / "split.csv")
            self.assertEqual(set(saved[saved.partition == "heldout"].benchmark_id), {"b"})
            self.assertEqual(set(saved[saved.partition == "train"].benchmark_id), {"a"})
            self.assertTrue((directory / "best_model.pt").is_file())
            with self.assertRaises(FileExistsError):
                training.run_fold(split, vectors, directory, settings, "test")


if __name__ == "__main__":
    unittest.main()
