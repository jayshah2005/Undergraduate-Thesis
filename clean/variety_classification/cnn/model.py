"""Small whole-leaf CNN that predicts a leaf's variety, on any number of input bands."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cnn_common import MaskedBatchNorm2d, min_pool  # noqa: E402

WIDTHS = (16, 24, 48)  # 1x1 spectral conv, block 1, blocks 2-3
N_VARIETIES = 4


class VarietyCNN(nn.Module):
    """1x1 spectral conv → 4 masked 3x3 conv layers (2 max-pools) → masked mean → variety logits.

    The same network for 204 bands and for RGB; only the input of the first layer changes,
    so a score difference comes from the bands. Mean pooling over leaf positions means
    leaves of any size give the same-length input to the last layer.

    With n_spectrum > 0 the last layer also reads the leaf's mean spectrum, directly: its
    logits are the 48-d part plus spectrum_weights times the spectrum, the same as one
    linear layer over both. Dropout applies to the 48-d part only.
    """

    def __init__(self, n_bands: int, dropout: float = 0.5, n_spectrum: int = 0):
        super().__init__()
        c0, c1, c2 = WIDTHS
        self.spec = nn.Conv2d(n_bands, c0, 1, bias=False)
        self.conv1 = nn.Conv2d(c0, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn = nn.ModuleList([MaskedBatchNorm2d(c) for c in (c0, c1, c2, c2, c2)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, N_VARIETIES))
        # Zeros draw no random numbers, so a seed starts the CNN part exactly as in the plain model.
        self.spectrum_weights = nn.Parameter(torch.zeros(n_spectrum, N_VARIETIES)) if n_spectrum else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, spectrum: torch.Tensor | None = None) -> torch.Tensor:
        """Score each leaf.

        B: batch size. d: input bands. S: canvas side.

        :param x: Preprocessed cubes, shape (B, d, S, S), 0 off the leaf.
        :type x: torch.Tensor
        :param mask: 1 on leaf pixels, shape (B, 1, S, S).
        :type mask: torch.Tensor
        :param spectrum: Standardised mean spectra, shape (B, d). None leaves out that part.
        :type spectrum: torch.Tensor | None
        :return: Variety logits, shape (B, 4).
        :rtype: torch.Tensor
        """
        h = F.relu(self.bn[0](self.spec(x), mask))
        h = F.relu(self.bn[1](self.conv1(h), mask))
        # A pooled position is leaf only if all four pixels under it are leaf.
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = F.relu(self.bn[2](self.conv2(h * mask), mask))
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = F.relu(self.bn[3](self.conv3(h * mask), mask))
        h = F.relu(self.bn[4](self.conv4(h), mask))

        # Positions whose 3x3 neighbourhood touches the background are left out, so the
        # leaf margin can't drive the score. A leaf too small to keep any uses its full mask.
        valid = min_pool(mask, 3, 1, pad=1)
        empty = valid.flatten(1).sum(1) == 0
        valid = torch.where(empty[:, None, None, None], mask, valid)
        z = (h * valid).sum((2, 3)) / valid.sum((2, 3))    # (B, 48)
        logits = self.head(z)
        return logits if spectrum is None else logits + spectrum @ self.spectrum_weights
