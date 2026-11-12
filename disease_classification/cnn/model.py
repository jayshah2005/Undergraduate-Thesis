"""Whole-leaf disease CNN: any backbone from helpers/backbones.py → masked mean or attention pooling → 1 logit."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers.backbones import DEFAULT_BACKBONE, SmallCNN, make_backbone  # noqa: E402
from helpers.cnn_common import LeafPool  # noqa: E402

WIDTHS = SmallCNN.widths   # the default backbone's layer widths, drawn by plot_architecture.py
# The best of the other backbones tested (chosen on the validation rows), reported beside the default.
SECOND_BACKBONE: str | None = "small_resnet"


class LeafCNN(nn.Module):
    """Backbone → masked pooling → dropout → one logit.

    With the default backbone every position of the last feature map stands for a ~26 px
    patch of leaf. Pooling turns any number of positions into one vector, so leaves of
    any size give the same-length input to the last layer.

    pooling="mean": every leaf position gets the same weight.
    pooling="attention": gated attention (Ilse et al. 2018) learns one weight per
    position, so a few symptomatic patches can decide the leaf.
    """

    def __init__(self, n_bands: int = 204, pooling: str = "attention", dropout: float = 0.5,
                 backbone: str = DEFAULT_BACKBONE):
        super().__init__()
        # Built in this order (backbone, pooling, head) so a seed gives the same weights as before the refactor.
        self.backbone = make_backbone(backbone, n_bands)
        self.pool = LeafPool(self.backbone.out_channels, pooling)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_channels, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score each leaf.

        B: batch size. S: canvas size. S4 = S / 4: size of the last feature map.

        :param x: Preprocessed cubes, shape (B, d, S, S) with d input bands, 0 off the leaf.
        :type x: torch.Tensor
        :param mask: 1 on leaf pixels, shape (B, 1, S, S).
        :type mask: torch.Tensor
        :return: logits (B,) and pooling weights (B, S4, S4) that sum to 1 per leaf.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        h, m = self.backbone(x, mask)
        z, a = self.pool(h, m)
        return self.head(z).squeeze(-1), a
