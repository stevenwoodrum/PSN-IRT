"""Proposal Approach 2 (scalar-ability variant): metadata/text-conditioned 2PL IRT.

P(correct) = sigmoid(a_i * (theta_s - b_i)), with scalar theta_s from a subject
network and (a_i, b_i) from an item network. The 4PL guessing (c) and
feasibility (d) parameters are omitted, as in Proposal_Team_5 Section III.
a_i = softplus(z_a) + A_FLOOR > 0, so probability is strictly increasing in
theta. Legacy Semantic_IRT and ordered_metadata_4pl_v1 checkpoints are not
accepted by this module's loader and keep their own meaning.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


MODEL_VERSION = "proposed_2pl_scalar_v1"
A1_SUBJECT_FIELDS = ("normalized_name", "provider", "reasoning_effort")
ALL_SUBJECT_FIELDS = A1_SUBJECT_FIELDS + (
    "release_date", "access_date", "harness", "harness_version", "subject_features_extra")
A_FLOOR = 0.1
PROBLEM_INDEX_CAP = 30


class CategoricalEncoder:
    """Field-wise one-hot: train-only categories, missing -> "", unknown -> zeros.

    Only fields that predict() receives in the subject record are accepted.
    subject_id, benchmark_id and outcomes can never be features here.
    """

    def __init__(self, categories):
        if not categories or any(k not in ALL_SUBJECT_FIELDS for k in categories):
            raise ValueError("Expected a nonempty subset of the predict() subject fields.")
        self.categories = {k: list(v) for k, v in categories.items()}
        self.lookup, offset = {}, 0
        for field, values in self.categories.items():
            if not values or len(set(values)) != len(values):
                raise ValueError("Categories must be nonempty and unique.")
            self.lookup[field] = {value: offset + i for i, value in enumerate(values)}
            offset += len(values)
        self.dimensions = offset

    @classmethod
    def fit(cls, frame, fields):
        if any(not isinstance(v, str) for f in fields for v in frame[f]):
            raise ValueError("Canonicalize missing metadata to strings first.")
        return cls({f: sorted(frame[f].unique()) for f in fields})

    def transform(self, frame):
        vectors = np.zeros((len(frame), self.dimensions), dtype=np.float32)
        for field, lookup in self.lookup.items():
            idx = frame[field].map(lookup).fillna(-1).to_numpy(dtype=int)
            known = idx >= 0
            vectors[np.flatnonzero(known), idx[known]] = 1
        return vectors


def item_feature_tokens(text):
    """Parse 'key=value;key=value' item_features into categorical tokens.

    problem_idx values >= PROBLEM_INDEX_CAP share one token (as in the
    November 7 diagnosis). Missing/blank features give no tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = []
    for part in text.split(";"):
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if key == "problem_idx":
            try:
                value = str(min(int(value), PROBLEM_INDEX_CAP))
            except ValueError:
                pass
        tokens.append(f"{key}={value}" if sep else key)
    return tokens


class TokenEncoder:
    """Multi-hot over a training-fitted token vocabulary; unknown tokens -> zeros."""

    def __init__(self, vocabulary):
        vocabulary = list(vocabulary)
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary must be unique.")
        self.vocabulary = vocabulary
        self.lookup = {t: i for i, t in enumerate(vocabulary)}
        self.dimensions = len(vocabulary)

    @classmethod
    def fit(cls, feature_strings):
        return cls(sorted({t for s in feature_strings for t in item_feature_tokens(s)}))

    def transform(self, feature_strings):
        vectors = np.zeros((len(feature_strings), self.dimensions), dtype=np.float32)
        for row, text in enumerate(feature_strings):
            for token in item_feature_tokens(text):
                if token in self.lookup:
                    vectors[row, self.lookup[token]] = 1
        return vectors


class ProposedIRT(nn.Module):
    """Scalar-ability 2PL with the same subject/item trunk shapes as FeatureIRT."""

    def __init__(self, subject_dim, item_dim, hidden_dim=64, dropout=0.):
        super().__init__()
        self.student_net = nn.Sequential(
            nn.Linear(subject_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.item_embed_proj = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.item_param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def item_parameters(self, items):
        raw = self.item_param_head(self.item_embed_proj(items))
        b = raw[:, 0]
        a = F.softplus(raw[:, 1]) + A_FLOOR
        return b, a

    def ability(self, subjects):
        return self.student_net(subjects).flatten()

    @staticmethod
    def logit(theta, b, a):
        return a * (theta - b)

    def logits(self, subjects, items):
        b, a = self.item_parameters(items)
        return self.logit(self.ability(subjects), b, a)

    def forward(self, subjects, items):
        return torch.sigmoid(self.logits(subjects, items))
