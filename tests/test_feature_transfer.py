import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feature_transfer import (DEFAULT_SETTINGS, Recipe, evaluate_fold, load_checkpoint,
                              make_tensors, predict, run_fold)
from models.feature_irt import FeatureIRT, MetadataEncoder
from models.Semantic_IRT import Semantic_IRT
from runtime import select_device


class FeatureTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_device_auto_uses_cpu_when_accelerators_are_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), \
                patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), "cpu")
        self.assertEqual(select_device("cpu"), "cpu")
        with self.assertRaises(ValueError):
            select_device("meta")

    def test_ordered_bounds_and_ability_monotonicity_at_extreme_logits(self):
        model = FeatureIRT(2, 4).eval()
        with torch.no_grad():
            model.item_param_head[-1].weight.zero_()
            for logits in ((0., -100., 100., -100.), (1., 100., -100., 100.), (0., 0., 0., 0.)):
                model.item_param_head[-1].bias.copy_(torch.tensor(logits))
                beta, a, c, d = model.item_parameters(torch.zeros(1, 4))
                self.assertTrue((a > 0).all() and (c >= 0).all() and (c <= d).all() and (d <= 1).all())
                p = model.probability(torch.linspace(-20, 20, 101), beta, a, c, d)
                self.assertTrue(torch.isfinite(p).all() and (p.diff() >= 0).all())
                self.assertTrue(((p >= c) & (p <= d)).all())

    def test_name_control_initial_weights_match_legacy_before_formula_change(self):
        torch.manual_seed(5814)
        old = Semantic_IRT(3, item_embed_dim=4)
        torch.manual_seed(5814)
        new = FeatureIRT(3, 4, mode="name_mean")
        # Insertion of parameter-free dropout changes Sequential indices only.
        old_params, new_params = list(old.parameters()), list(new.parameters())
        self.assertEqual(len(old_params), len(new_params))
        for left, right in zip(old_params, new_params):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_train_only_categories_missing_and_unseen_fields(self):
        train = pd.DataFrame(dict(normalized_name=["a", "b"], provider=["p", "q"], reasoning_effort=["", "high"]))
        encoder = MetadataEncoder.fit(train)
        heldout = pd.DataFrame(dict(normalized_name=["unseen", "a"], provider=["p", "never"], reasoning_effort=["", "new"]))
        vectors = encoder.transform(heldout)
        self.assertNotIn("unseen", encoder.categories["normalized_name"])
        self.assertEqual(vectors[0].sum(), 2)  # provider and observed missingness survive.
        self.assertEqual(vectors[1].sum(), 1)
        self.assertEqual(vectors[0, encoder.lookup["reasoning_effort"][""]], 1)

    def test_metadata_uses_provider_when_name_is_unseen(self):
        model = FeatureIRT(3, 4).eval()
        with torch.no_grad():
            model.student_net[0].weight[:, 1].fill_(0)
            model.student_net[0].weight[0, 1] = 5
            absent = model.ability(torch.zeros(1, 3))
            available = model.ability(torch.tensor([[0., 1., 0.]]))
        self.assertNotAlmostEqual(absent.item(), available.item(), places=5)

    def test_training_reload_export_and_unseen_rows_agree(self):
        frame = pd.DataFrame([dict(row_id=i, benchmark_id="b", subject_id="s"+str(i%2),
            item_id="i"+str(i), response_id=str(i), item_group="g"+str(i),
            content="prompt "+str(i), normalized_name="a" if i < 8 else "new",
            provider="p", reasoning_effort="", response=i%2) for i in range(12)])
        split = dict(train=frame.iloc[:6], validation=frame.iloc[6:8], heldout=frame.iloc[8:], purged=frame.iloc[:0])
        vectors = {item: np.array([1, .5, 0, .2], dtype=np.float32) for item in frame.item_id}
        recipe = Recipe("test")
        settings = {**DEFAULT_SETTINGS, "epochs": 2, "batch_size": 4}
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold"
            run_fold(split, vectors, root, settings, recipe, "test", device="cpu")
            # Outer rows really are withheld until the independent scoring stage.
            first = pd.read_csv(root / "predictions.csv", keep_default_na=False)
            self.assertNotIn("heldout", set(first.partition))
            model, encoder, saved_recipe, checkpoint = load_checkpoint(root / "best_model.pt")
            evaluate_fold(model, encoder, saved_recipe, split, vectors, root, "test", checkpoint["best_epoch"], "cpu")
            saved = pd.read_csv(root / "predictions.csv", keep_default_na=False)
            actual = predict(model, make_tensors(frame, vectors, encoder, recipe, "cpu"))
            np.testing.assert_allclose(saved.probability, actual, atol=1e-7)
            self.assertEqual((~saved[saved.partition == "heldout"].known_subject).sum(), 4)
            abilities = pd.read_csv(root / "subject_abilities.csv", keep_default_na=False)
            params = pd.read_csv(root / "item_parameters.csv", keep_default_na=False).set_index("item_id")
            # Reconstruct probabilities from exported numeric parameters, not model code.
            joined = saved.merge(abilities, on=["normalized_name", "provider", "reasoning_effort"], validate="many_to_one")
            beta, a, c, d = [params.loc[joined.item_id, field].to_numpy() for field in ("beta", "a", "c", "d")]
            expected = c + (d-c)/(1+np.exp(-a*(joined.ability.to_numpy()-beta)))
            np.testing.assert_allclose(joined.probability, expected, atol=1e-7)
            history = pd.read_csv(root / "training_history.csv")
            self.assertEqual(checkpoint["best_epoch"], int(history.loc[history.validation_equal_pair_Brier.idxmin(), "epoch"]))
            self.assertEqual(encoder.categories["reasoning_effort"], [""])
            with self.assertRaises(FileExistsError):
                run_fold(split, vectors, root, settings, recipe, "test", device="cpu")
            bad = root / "legacy.pt"
            torch.save({"state_dict": model.state_dict()}, bad)
            with self.assertRaises(ValueError):
                load_checkpoint(bad)

    def test_no_text_control_has_identical_input_for_different_items(self):
        frame = pd.DataFrame(dict(item_id=["a", "b"], normalized_name=["n", "n"],
                                  provider=["p", "p"], reasoning_effort=["", ""], response=[0, 1]))
        encoder = MetadataEncoder.fit(frame)
        data = make_tensors(frame, {"a": np.ones(4), "b": np.zeros(4)}, encoder, Recipe("no_text", text=False), "cpu")
        np.testing.assert_array_equal(data[1], [[1], [1]])


if __name__ == "__main__":
    unittest.main()
