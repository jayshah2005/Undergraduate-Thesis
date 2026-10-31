"""Small whole-leaf CNN with masked normalisation and masked mean or attention pooling."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cnn_common import MaskedBatchNorm2d, min_pool  # noqa: E402

WIDTHS = (16, 24, 48)  # 1x1 spectral conv, block 1, blocks 2-3
ATTN_HIDDEN = 32


class LeafCNN(nn.Module):
    """1x1 spectral conv → 4 masked 3x3 conv layers (2 max-pools) → masked pooling → logit.

    Every position of the last feature map stands for a ~26 px patch of leaf (the
    receptive field). Pooling turns any number of positions into one 48-number
    vector, so leaves of any size give the same-length input to the last layer.

    pooling="mean": every leaf position gets the same weight.
    pooling="attention": gated attention (Ilse et al. 2018) learns one weight per
    position, so a few symptomatic patches can decide the leaf.
    """

    def __init__(self, n_bands: int = 204, pooling: str = "attention", dropout: float = 0.5):
        super().__init__()
        assert pooling in ("mean", "attention")
        c0, c1, c2 = WIDTHS
        self.pooling = pooling
        self.spec = nn.Conv2d(n_bands, c0, 1, bias=False)   # learned band reduction
        self.conv1 = nn.Conv2d(c0, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn = nn.ModuleList([MaskedBatchNorm2d(c) for c in (c0, c1, c2, c2, c2)])
        if pooling == "attention":
            self.att_v = nn.Linear(c2, ATTN_HIDDEN)
            self.att_u = nn.Linear(c2, ATTN_HIDDEN)
            self.att_w = nn.Linear(ATTN_HIDDEN, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, 1))

    def _block(self, conv, bn, h, mask):
        """Conv → masked BN → ReLU; the background stays 0."""
        return F.relu(bn(conv(h), mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score each leaf.

        B: batch size. S: canvas size. S4 = S / 4: size of the last feature map.

        :param x: Preprocessed cubes, shape (B, 204, S, S), 0 off the leaf.
        :type x: torch.Tensor
        :param mask: 1 on leaf pixels, shape (B, 1, S, S).
        :type mask: torch.Tensor
        :return: logits (B,) and pooling weights (B, S4, S4) that sum to 1 per leaf.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        h = self._block(self.spec, self.bn[0], x, mask)
        h = self._block(self.conv1, self.bn[1], h, mask)
        # A pooled position is leaf only if all four pixels under it are leaf.
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = self._block(self.conv2, self.bn[2], h * mask, mask)
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = self._block(self.conv3, self.bn[3], h * mask, mask)
        h = self._block(self.conv4, self.bn[4], h, mask)

        # Positions whose 3x3 neighbourhood touches the background are left out, so the
        # leaf margin cannot drive the prediction. Fall back to the full mask for a leaf
        # too small to keep any position after that.
        valid = min_pool(mask, 3, 1, pad=1)
        empty = valid.flatten(1).sum(1) == 0
        valid = torch.where(empty[:, None, None, None], mask, valid)

        b, c, hh, ww = h.shape
        feats = h.flatten(2).transpose(1, 2)                  # (B, positions, C)
        keep = valid.flatten(1) > 0                           # (B, positions)
        if self.pooling == "attention":
            s = self.att_w(torch.tanh(self.att_v(feats)) * torch.sigmoid(self.att_u(feats)))
            s = s.squeeze(-1).masked_fill(~keep, float("-inf"))
            a = torch.softmax(s, dim=1)
        else:
            a = keep.float() / keep.float().sum(1, keepdim=True)
        z = (a.unsqueeze(-1) * feats).sum(1)                  # (B, C)
        return self.head(z).squeeze(-1), a.view(b, hh, ww)