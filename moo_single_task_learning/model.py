"""Shared trunk + attention pool + six binary heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same names as data.py (redefined here; not imported).
N_HEADS = 6


class AttentionPool(nn.Module):
    """Gated attention over leaf pixels → one vector per leaf (Ilse et al. 2018).

    Instead of averaging every leaf pixel equally, the net learns a weight per
    position so a few spots (e.g. a blotch) can dominate the leaf summary.
    """

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.att_v = nn.Linear(channels, hidden)
        self.att_u = nn.Linear(channels, hidden)
        self.att_w = nn.Linear(hidden, 1)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param h: Features (B, C, H, W).
        :param mask: Leaf mask (B, 1, H, W).
        :return: z (B, C) and attention map (B, H, W).
        """
        b, c, hh, ww = h.shape
        # List of position vectors: (B, N, C) with N = H*W.
        feats = h.flatten(2).transpose(1, 2)
        keep = mask.flatten(1) > 0

        # Gated score per position, then softmax (background → -inf → weight 0).
        s = self.att_w(
            torch.tanh(self.att_v(feats)) * torch.sigmoid(self.att_u(feats)),
        )
        s = s.squeeze(-1).masked_fill(~keep, float("-inf"))
        a = torch.softmax(s, dim=1)
        z = (a.unsqueeze(-1) * feats).sum(1)
        return z, a.view(b, hh, ww)


class SixHeadCNN(nn.Module):
    """1x1 spectral mix → 3x3 convs (no max-pool) → attention → 6 logits."""

    def __init__(self, n_bands: int = 204, width: int = 32):
        super().__init__()
        self.spec = nn.Conv2d(n_bands, width, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.pool = AttentionPool(width)
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(N_HEADS)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: SNV cubes (B, 204, S, S).
        :param mask: (B, 1, S, S).
        :return: Logits (B, 6) — variety heads 0..3, RB=4, LR=5.
        """
        h = F.relu(self.spec(x))
        h = F.relu(self.conv1(h)) * mask
        h = F.relu(self.conv2(h)) * mask
        z, _att = self.pool(h, mask)
        return torch.cat([head(z) for head in self.heads], dim=1)


if __name__ == "__main__":
    from data import fold_split, get_device, load_vines, make_batch

    device = get_device()
    data = load_vines()
    _tr, _va, te = fold_split(data.fold_ids, 0)
    x, mask, y = make_batch(data, te[:4], device)

    model = SixHeadCNN().to(device)
    logits = model(x, mask)
    print("logits", tuple(logits.shape), "Y", tuple(y.shape))
    print("logits[0]", logits[0].detach().tolist())