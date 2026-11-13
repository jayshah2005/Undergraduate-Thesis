"""Variety-conditioned leaf CNN: the repo's small CNN with FiLM after every batch norm.

FiLM (Perez, Strub, de Vries, Dumoulin & Courville 2018, "FiLM: Visual Reasoning with a General Conditioning
Layer", AAAI 32(1), arXiv 1709.07871): each feature map h of layer l becomes gamma_(v,l) * h + beta_(v,l),
with (gamma, beta) per channel generated from the condition, here the leaf's variety v. As in the paper, the
modulation sits after batch norm and before the ReLU, and gamma is parameterised as 1 + delta.
Adaptations: the condition is a 3-way variety one-hot, so the FiLM generator is one embedding per variety
(a linear layer on a one-hot); it starts at the identity (delta = beta = 0), so training begins from the
plain pooled CNN. All filters stay shared across varieties; only the per-channel scale and shift differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]
for p in (R, R / "disease_classification/cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.backbones import SmallCNN, min_pool  # noqa: E402
from model import LeafCNN  # noqa: E402


class FiLMLeafCNN(nn.Module):
    """LeafCNN(small_cnn) whose 5 batch norms are each followed by a per-variety FiLM layer."""

    def __init__(self, n_bands: int, pooling: str, dropout: float, n_varieties: int = 3):
        super().__init__()
        self.net = LeafCNN(n_bands, pooling, dropout, "small_cnn")     # same weights for a seed as the plain CNN
        assert isinstance(self.net.backbone, SmallCNN)
        c0, c1, c2 = SmallCNN.widths
        self.chans = (c0, c1, c2, c2, c2)
        self.film = nn.Embedding(n_varieties, 2 * sum(self.chans))
        nn.init.zeros_(self.film.weight)                                  # identity at the start

    def _mod(self, h, params, layer, mask):
        """FiLM on leaf positions only: the shift must not light up the background (the backbone keeps it 0)."""
        lo = 2 * sum(self.chans[:layer])
        c = self.chans[layer]
        d_gamma, beta = params[:, lo:lo + c], params[:, lo + c:lo + 2 * c]
        return (h * (1 + d_gamma)[:, :, None, None] + beta[:, :, None, None]) * mask

    def forward(self, x, mask, variety):
        b, p = self.net.backbone, self.film(variety)
        h = F.relu(self._mod(b.bn[0](b.spec(x), mask), p, 0, mask))
        h = F.relu(self._mod(b.bn[1](b.conv1(h), mask), p, 1, mask))
        h, m = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = F.relu(self._mod(b.bn[2](b.conv2(h * m), m), p, 2, m))
        h, m = F.max_pool2d(h, 2), min_pool(m, 2, 2)
        h = F.relu(self._mod(b.bn[3](b.conv3(h * m), m), p, 3, m))
        h = F.relu(self._mod(b.bn[4](b.conv4(h), m), p, 4, m))
        z, a = self.net.pool(h, m)
        return self.net.head(z).squeeze(-1), a
