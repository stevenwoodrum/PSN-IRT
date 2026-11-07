"""Versioned, ordered 4PL experiment; legacy checkpoints keep their old meaning."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SUBJECT_FIELDS = ("normalized_name", "provider", "reasoning_effort")
MODEL_VERSION = "ordered_metadata_4pl_v1"


class MetadataEncoder:
    """A1's field-wise one-hot policy: train-only categories, unknown -> zeros.

    Missing values are canonicalized to the empty string before reaching this
    class. A missing value observed in training is a learned category, just as
    in A1. No subject ID, benchmark ID or outcome enters these features.
    """

    def __init__(self, categories):
        if not categories or any(k not in SUBJECT_FIELDS for k in categories):
            raise ValueError("Expected a nonempty subset of the three A1 fields.")
        self.categories = categories
        self.lookup, offset = {}, 0
        for field, values in categories.items():
            if not values or len(set(values)) != len(values):
                raise ValueError("Categories must be nonempty and unique.")
            self.lookup[field] = {value: offset + i for i, value in enumerate(values)}
            offset += len(values)
        self.dimensions = offset

    @classmethod
    def fit(cls, frame, fields=SUBJECT_FIELDS):
        if any(not isinstance(value, str) for field in fields for value in frame[field]):
            raise ValueError("Canonicalize missing metadata to string values first.")
        return cls({field: sorted(frame[field].unique()) for field in fields})

    def transform(self, frame):
        vectors = np.zeros((len(frame), self.dimensions), dtype=np.float32)
        for field, lookup in self.lookup.items():
            indices = frame[field].map(lookup).fillna(-1).to_numpy(dtype=int)
            known = indices >= 0
            vectors[np.flatnonzero(known), indices[known]] = 1
        return vectors


class FeatureIRT(nn.Module):
    """Scalar ability conditioned on metadata, with ordered lower/upper bounds.

    ``name_mean`` preserves the legacy unknown-name mean-ability rule for the
    corrected-formula control. ``metadata`` uses every available field even
    when the name is unseen; an all-unknown record has the A1 all-zero input.
    """

    def __init__(self, subject_dim, item_dim, hidden_dim=64, dropout=0., mode="metadata"):
        super().__init__()
        if mode not in ("name_mean", "metadata"):
            raise ValueError("Unknown subject mode.")
        self.mode = mode
        self.subject_dim = subject_dim
        # Keep the baseline's initialization order and RNG draws at dropout=0.
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
            nn.Dropout(dropout), nn.Linear(hidden_dim, 4),
        )

    def item_parameters(self, item_vectors):
        raw = self.item_param_head(self.item_embed_proj(item_vectors))
        beta = raw[:, 0]
        a = F.softplus(raw[:, 1]) + .1
        c = .3 * torch.sigmoid(raw[:, 2])
        d = c + (1 - c) * torch.sigmoid(raw[:, 3])
        return beta, a, c, d

    def ability(self, subject_vectors):
        theta = self.student_net(subject_vectors).flatten()
        if self.mode == "name_mean":
            identities = torch.eye(self.subject_dim, device=theta.device)
            mean = self.student_net(identities).mean()
            theta = torch.where(subject_vectors.sum(1) == 0, mean, theta)
        return theta

    @staticmethod
    def probability(theta, beta, a, c, d):
        return c + (d - c) * torch.sigmoid(a * (theta - beta))

    def forward(self, subjects, items):
        return self.probability(self.ability(subjects), *self.item_parameters(items))

    def kernel_penalty(self):
        # Explicit kernel L2, distinct from AdamW's decoupled weight decay.
        return sum(layer.weight.square().sum() for layer in self.modules()
                   if isinstance(layer, nn.Linear))
