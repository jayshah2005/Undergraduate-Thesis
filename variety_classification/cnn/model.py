"""Whole-leaf CNN that predicts a leaf's variety: any backbone from helpers/backbones.py → masked mean → 4 logits."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.backbones import DEFAULT_BACKBONE, SmallCNN, make_backbone  # noqa: E402
from helpers.cnn_common import LeafPool  # noqa: E402

WIDTHS = SmallCNN.widths   # the default backbone's layer widths, drawn by plot_architecture.py
N_VARIETIES = 4
# The best of the other backbones tested (chosen on the validation rows), reported beside the default.
SECOND_BACKBONE: str | None = "small_resnet"


class VarietyCNN(nn.Module):
    """Backbone → masked mean over leaf positions → dropout → linear layer with 4 outputs.

    The same network for 204 bands and for RGB; only the input of the first layer changes,
    so a score difference comes from the bands. Mean pooling over leaf positions means
    leaves of any size give the same-length input to the last layer.
    """

    def __init__(self, n_bands: int, dropout: float = 0.5, backbone: str = DEFAULT_BACKBONE):
        super().__init__()
        # Built in this order (backbone, pooling, head) so a seed gives the same weights as before the refactor.
        self.backbone = make_backbone(backbone, n_bands)
        self.pool = LeafPool(self.backbone.out_channels, "mean")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_channels, N_VARIETIES))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score each leaf.

        B: batch size. d: input bands. S: canvas side.

        :param x: Preprocessed cubes, shape (B, d, S, S), 0 off the leaf.
        :type x: torch.Tensor
        :param mask: 1 on leaf pixels, shape (B, 1, S, S).
        :type mask: torch.Tensor
        :return: Variety logits, shape (B, 4).
        :rtype: torch.Tensor
        """
        h, m = self.backbone(x, mask)
        z, _weights = self.pool(h, m)
        return self.head(z)
