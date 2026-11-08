from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.proposed_irt import (A_FLOOR, ALL_SUBJECT_FIELDS, CategoricalEncoder, ProposedIRT,
                                 TokenEncoder, item_feature_tokens)
from proposed_training import (DEFAULT_SETTINGS, RECIPES, Recipe, evaluate_fold, load_checkpoint,
                               make_data, predict, run_fold, within_pair_auc)


def toy_frame(n=16):
    rows = []
    for i in range(n):
        rows.append(dict(row_id=i, benchmark_id="b", subject_id="s" + str(i % 3), item_id="i" + str(i),
                         response_id=str(i), item_group="g" + str(i), content="prompt " + str(i),
                         normalized_name="n" + str(i % 3) if i < 12 else "new", provider="p",
                         reasoning_effort="", release_date="2025-01-0" + str(i % 3 + 1), access_date="",
                         harness="h" + str(i % 2), harness_version="", subject_features_extra="x" + str(i % 3),
                         item_features="competition=c%d;problem_idx=%d" % (i % 2, i), response=i % 2))
    return pd.DataFrame(rows)


class ProposedIRTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_2pl_is_monotone_bounded_and_has_no_c_or_d(self):
        model = ProposedIRT(2, 4).eval()
        self.assertEqual(model.item_param_head[-1].out_features, 2)
        with torch.no_grad():
            model.item_param_head[-1].weight.zero_()
            for logits in ((0., -100.), (3., 100.), (-2., 0.)):
                model.item_param_head[-1].bias.copy_(torch.tensor(logits))
                b, a = model.item_parameters(torch.zeros(1, 4))
                self.assertTrue((a >= A_FLOOR).all())
                theta = torch.linspace(-20, 20, 101)
                p = torch.sigmoid(model.logit(theta, b, a))
                self.assertTrue(torch.isfinite(p).all() and (p.diff() >= 0).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                # 2PL midpoint: theta == b gives exactly 0.5.
                self.assertAlmostEqual(torch.sigmoid(model.logit(b, b, a)).item(), .5, places=6)

    def test_bce_with_logits_matches_probability_bce(self):
        z = torch.tensor([-3., 0., 2.5]); y = torch.tensor([0., 1., 1.])
        torch.testing.assert_close(F.binary_cross_entropy_with_logits(z, y),
                                   F.binary_cross_entropy(torch.sigmoid(z), y))

    def test_item_feature_tokens_and_train_only_vocabulary(self):
        self.assertEqual(item_feature_tokens("competition=aime_2025;problem_idx=2;prompt_source=user_message"),
                         ["competition=aime_2025", "problem_idx=2", "prompt_source=user_message"])
        self.assertEqual(item_feature_tokens("problem_idx=47"), ["problem_idx=30"])
        self.assertEqual(item_feature_tokens(""), [])
        self.assertEqual(item_feature_tokens(None), [])
        encoder = TokenEncoder.fit(["lang=c++", "lang=java", ""])
        vectors = encoder.transform(["lang=rust", "lang=java", ""])
        self.assertEqual(vectors[0].sum(), 0)  # unseen value -> zeros
        self.assertEqual(vectors[1, encoder.lookup["lang=java"]], 1)
        self.assertEqual(vectors[2].sum(), 0)

    def test_subject_encoder_rejects_ids_and_zeroes_unknowns(self):
        with self.assertRaises(ValueError):
            CategoricalEncoder({"subject_id": ["a"]})
        with self.assertRaises(ValueError):
            CategoricalEncoder({"benchmark_id": ["a"]})
        frame = toy_frame()
        encoder = CategoricalEncoder.fit(frame.iloc[:6], ALL_SUBJECT_FIELDS)
        unseen = frame.iloc[[12]].assign(normalized_name="zzz", release_date="2099", harness="new", subject_features_extra="new")
        vector = encoder.transform(unseen)[0]
        # provider, reasoning_effort (""), access_date (""), harness_version ("") remain known.
        self.assertEqual(vector.sum(), 4)

    def test_all_eight_fields_resolve_more_identities_than_a1_fields(self):
        frame = toy_frame()
        a1 = len(frame[["normalized_name", "provider", "reasoning_effort"]].drop_duplicates())
        full = len(frame[list(ALL_SUBJECT_FIELDS)].drop_duplicates())
        self.assertGreaterEqual(full, a1)
        self.assertEqual([r.name for r in RECIPES],
                         ["2pl_a1_fields", "2pl_all_subject_fields", "2pl_item_features", "2pl_full"])

    def test_recipe_input_dimensions(self):
        frame = toy_frame()
        vectors = {i: np.ones(4, dtype=np.float32) for i in frame.item_id}
        for recipe, expected in ((Recipe("t"), 4), (Recipe("f", item_features=True), 4 + 2 + 12)):
            from proposed_training import fit_encoders
            subject, tokens = fit_encoders(frame.iloc[:12], recipe)
            data = make_data(frame, vectors, subject, tokens, recipe, "cpu")
            self.assertEqual(data["items"].shape[1], expected)
            self.assertEqual(len(data["s_idx"]), len(frame))

    def test_within_pair_auc(self):
        frame = pd.DataFrame(dict(benchmark_id=["b"]*4 + ["c"]*2, subject_id=["s"]*4 + ["t"]*2,
                                  response=[0, 1, 0, 1, 1, 1]))
        auc, pairs = within_pair_auc(frame, [.1, .9, .2, .8, .5, .5])
        self.assertEqual((auc, pairs), (1.0, 1))

    def test_training_reload_export_and_outer_withholding(self):
        frame = toy_frame()
        split = dict(train=frame.iloc[:8], validation=frame.iloc[8:12], heldout=frame.iloc[12:], purged=frame.iloc[:0])
        vectors = {item: np.array([1, .5, 0, .2], dtype=np.float32) * (int(item[1:]) % 3) for item in frame.item_id}
        recipe = Recipe("test", ALL_SUBJECT_FIELDS, item_features=True)
        settings = {**DEFAULT_SETTINGS, "epochs": 3, "batch_size": 4}
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "fold"
            run_fold(split, vectors, root, settings, recipe, "test", device="cpu")
            first = pd.read_csv(root / "predictions.csv", keep_default_na=False)
            self.assertNotIn("heldout", set(first.partition))
            model, subject, tokens, saved_recipe, checkpoint = load_checkpoint(root / "best_model.pt")
            self.assertEqual(saved_recipe, recipe)
            # Vocabulary comes from training items only.
            self.assertEqual(tokens.vocabulary, sorted({t for s in split["train"].item_features for t in item_feature_tokens(s)}))
            evaluate_fold(model, subject, tokens, saved_recipe, split, vectors, root, "test", checkpoint["best_epoch"], "cpu")
            saved = pd.read_csv(root / "predictions.csv", keep_default_na=False, dtype={f: str for f in ALL_SUBJECT_FIELDS})
            actual = predict(model, make_data(frame, vectors, subject, tokens, recipe, "cpu"))
            np.testing.assert_allclose(saved.probability, actual, atol=1e-7)
            self.assertEqual((~saved[saved.partition == "heldout"].known_subject).sum(), 4)
            abilities = pd.read_csv(root / "subject_abilities.csv", keep_default_na=False, dtype=str)
            abilities["ability"] = abilities.ability.astype(float)
            params = pd.read_csv(root / "item_parameters.csv", keep_default_na=False).set_index("item_id")
            joined = saved.merge(abilities, on=list(ALL_SUBJECT_FIELDS), validate="many_to_one")
            b, a = params.loc[joined.item_id, "b"].to_numpy(), params.loc[joined.item_id, "a"].to_numpy()
            expected = 1 / (1 + np.exp(-a * (joined.ability.to_numpy() - b)))
            np.testing.assert_allclose(joined.probability, expected, atol=1e-6)
            history = pd.read_csv(root / "training_history.csv")
            self.assertEqual(checkpoint["best_epoch"], int(history.loc[history.validation_equal_pair_Brier.idxmin(), "epoch"]))
            with self.assertRaises(FileExistsError):
                run_fold(split, vectors, root, settings, recipe, "test", device="cpu")
            bad = root / "other.pt"
            torch.save({"model_version": "ordered_metadata_4pl_v1", "state_dict": model.state_dict()}, bad)
            with self.assertRaises(ValueError):
                load_checkpoint(bad)


if __name__ == "__main__":
    unittest.main()
