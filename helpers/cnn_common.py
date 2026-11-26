"""Masked layers and canvas helpers shared by the whole-leaf CNNs."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBatchNorm2d(nn.Module):
    """Batch norm whose statistics use leaf positions only.

    Ordinary BatchNorm (and GroupNorm) would average over the zero padding too, so
    the statistics would depend on how much padding a leaf has, i.e. on leaf size.
    Here mean and variance are taken over positions where mask == 1, in fp32, and
    the output is multiplied by the mask so the background stays exactly 0.
    """

    def __init__(self, c: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalise x over leaf positions.

        B: batch size. C: channels. H, W: map size.

        :param x: Features, shape (B, C, H, W).
        :type x: torch.Tensor
        :param mask: 1 on leaf positions, 0 on padding, shape (B, 1, H, W).
        :type mask: torch.Tensor
        :return: Normalised features, shape (B, C, H, W), 0 off the leaf.
        :rtype: torch.Tensor
        """
        x = x.float()
        if self.training:
            n = mask.sum().clamp_min(2.0)
            mean = (x * mask).sum((0, 2, 3)) / n
            var = (((x - mean[None, :, None, None]) ** 2) * mask).sum((0, 2, 3)) / n
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return (y * self.weight[None, :, None, None] + self.bias[None, :, None, None]) * mask


def min_pool(mask: torch.Tensor, k: int, stride: int, pad: int = 0) -> torch.Tensor:
    """Erode a 0/1 mask: a position stays 1 only if its whole k×k window is 1.

    :param mask: 0/1 mask, shape (B, 1, H, W).
    :type mask: torch.Tensor
    :param k: Window size.
    :type k: int
    :param stride: Stride.
    :type stride: int
    :param pad: Zero padding on every side (outside the canvas counts as background).
    :type pad: int
    :return: Eroded mask, shape (B, 1, H', W').
    :rtype: torch.Tensor
    """
    if pad:
        mask = F.pad(mask, (pad, pad, pad, pad), value=0.0)
    return -F.max_pool2d(-mask, k, stride)


def pad_to_canvas(cubes: list[np.ndarray], device: torch.device,
                  side: int | None = None, align: int = 4) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Centre every leaf on one square canvas; background and padding are 0.

    By default the canvas side is the largest leaf height or width, rounded up to a
    multiple of align (the backbone's total stride: 4 for two 2x2 pools). Square so a 90°
    rotation keeps the leaf on the canvas. Offsets are multiples of align too, so the
    pooling grid lines up the same way for every leaf.

    n: the number of leaves. S: canvas side.

    :param cubes: n cubes, each (204, lines, samples) float32, background 0.
    :type cubes: list[np.ndarray]
    :param device: Where to keep the canvas.
    :type device: torch.device
    :param side: Fixed canvas side (a multiple of align, at least the largest leaf); None fits these leaves.
    :type side: int | None
    :param align: The backbone's total stride; canvas side and leaf offsets are multiples of it.
    :type align: int
    :return: X (n, 204, S, S) float16 reflectance, mask (n, 1, S, S) float32, S.
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    fit = -(-max(max(c.shape[1], c.shape[2]) for c in cubes) // align) * align
    if side is None:
        side = fit
    elif side < fit or side % align:
        raise ValueError(f"canvas {side} must be a multiple of {align} and at least {fit}")
    X = torch.zeros(len(cubes), cubes[0].shape[0], side, side, dtype=torch.float16, device=device)
    for i, c in enumerate(cubes):
        _, h, w = c.shape
        top, left = ((side - h) // 2) // align * align, ((side - w) // 2) // align * align
        X[i, :, top:top + h, left:left + w] = torch.from_numpy(c).to(device, torch.float16)
    mask = (X != 0).any(dim=1, keepdim=True).float()
    return X, mask, side


def augment(x: torch.Tensor, m: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random 90° rotations and flips, the same for a leaf's cube and mask.

    Rotation by 90° maps the square canvas onto itself. A moved leaf can sit off the pooling
    grid by up to the backbone's stride − 1 px (3 px; training only, test leaves stay on it),
    which adds a little variety.

    :param x: Input, shape (B, d, S, S).
    :type x: torch.Tensor
    :param m: Mask, shape (B, 1, S, S).
    :type m: torch.Tensor
    :param g: Seeded generator.
    :type g: torch.Generator
    :return: Augmented x and mask, same shapes.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    k = torch.randint(0, 4, (x.shape[0],), generator=g, device=x.device).tolist()
    flip = torch.rand(x.shape[0], generator=g, device=x.device).lt(0.5).tolist()
    xs, ms = [], []
    for i in range(x.shape[0]):
        xi, mi = torch.rot90(x[i], k[i], (1, 2)), torch.rot90(m[i], k[i], (1, 2))
        if flip[i]:
            xi, mi = xi.flip(2), mi.flip(2)
        xs.append(xi)
        ms.append(mi)
    return torch.stack(xs), torch.stack(ms)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average each channel over leaf positions only, so the padding never counts.

    B: batch size. C: channels. H, W: map size.

    :param h: Features, shape (B, C, H, W).
    :type h: torch.Tensor
    :param mask: 1 on the positions to average, shape (B, 1, H, W).
    :type mask: torch.Tensor
    :return: Per-channel means, shape (B, C); 0 for a leaf with no position.
    :rtype: torch.Tensor
    """
    return (h * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1.0)


def masked_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Take each channel's largest value over leaf positions only.

    Off-leaf positions are set to -inf before the max, not multiplied by the mask: features can
    be negative (e.g. a ResNet branch before its ReLU), and a 0 from the padding would then win.

    B: batch size. C: channels. H, W: map size.

    :param h: Features, shape (B, C, H, W).
    :type h: torch.Tensor
    :param mask: 1 on the positions to search, shape (B, 1, H, W).
    :type mask: torch.Tensor
    :return: Per-channel maxima, shape (B, C); 0 for a leaf with no position (never -inf).
    :rtype: torch.Tensor
    """
    m = h.masked_fill(mask == 0, float("-inf")).amax((2, 3))
    has_leaf = mask.sum((2, 3)) > 0                     # (B, 1), broadcast over channels
    return torch.where(has_leaf, m, torch.zeros_like(m))


def eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """ECA's kernel size k = |log2(C) / gamma + b / gamma|_odd (Wang et al. 2020, Eq. 12; gamma 2, b 1).

    |t|_odd is the nearest odd number to t, computed as in the authors' code. It gives 3 for every
    width in these backbones (16 to 64 channels).
    """
    t = int(abs((math.log2(channels) + b) / gamma))
    return t if t % 2 else t + 1


class MaskedECA(nn.Module):
    """Efficient Channel Attention: masked average → 1D conv across channels → sigmoid → rescale.

    Wang et al. 2020, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks",
    CVPR. One k-wide 1D conv, shared by all channels and without a bias (as in the authors' code),
    turns the C channel averages into C weights, so each channel's weight depends on its k
    neighbours; there is no dimension reduction. Changed for whole leaves: the global average is
    over leaf positions only (masked_mean), so a leaf's padding cannot change its weights.
    """

    def __init__(self, channels: int):
        super().__init__()
        k = eca_kernel_size(channels)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reweight each channel of h (B, C, H, W) by its learned weight; the background stays 0."""
        y = masked_mean(h, mask)                              # (B, C) channel descriptors
        w = torch.sigmoid(self.conv(y[:, None, :]))[:, 0]     # (B, C) channel weights in (0, 1)
        return h * w[:, :, None, None]


class MaskedCBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention, then spatial attention.

    Woo et al. 2018, "CBAM: Convolutional Block Attention Module", ECCV, Eqs. 1-3:
    - channel: a shared MLP (C → C/r → C, ReLU between) on the average- and the max-pooled
      descriptor, summed, sigmoid; r = 16 as in the paper;
    - spatial: the channel-wise average and max at each position → a 7×7 conv → sigmoid.
    Channel first, then spatial, in sequence (the paper's best order).

    Changed for whole leaves:
    - the average and max over positions use leaf positions only (masked_mean, masked_max);
    - with r = 16 the narrow layers here (16-64 channels) would get 1-4 hidden units, so the
      hidden width is floored at 1;
    - no batch norm after the 7×7 conv: the paper's text has none (the authors' code adds one).
    The 7×7 conv sees the zero background next to the leaf margin, the same zeros on any canvas,
    so a leaf's padding still cannot change its score; the output is re-masked to keep the
    background exactly 0.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reweight h (B, C, H, W) by channel, then by position; the background stays 0."""
        # Channel attention (Eq. 2): one weight per channel from the leaf's average and max.
        mc = torch.sigmoid(self.mlp(masked_mean(h, mask)) + self.mlp(masked_max(h, mask)))
        h = h * mc[:, :, None, None]
        # Spatial attention (Eq. 3): one weight per position from its average and max over channels.
        s = torch.cat([h.mean(1, keepdim=True), h.amax(1, keepdim=True)], 1)
        return h * torch.sigmoid(self.spatial(s)) * mask


BLOCK_ATTENTION_MODULES = {"eca": MaskedECA, "cbam": MaskedCBAM}


def make_block_attention(kind: str | None, channels: int) -> nn.Module | None:
    """Build an ECA or CBAM module for a layer with this many channels, or None for no module.

    :param kind: "eca", "cbam" or None.
    :type kind: str | None
    :param channels: Channels of the feature map it reweights.
    :type channels: int
    :return: A module mapping (h (B, C, H, W), mask (B, 1, H, W)) to a reweighted h, or None.
    :rtype: nn.Module | None
    """
    return None if kind is None else BLOCK_ATTENTION_MODULES[kind](channels)


class LeafPool(nn.Module):
    """Turn a feature map into one vector per leaf: masked mean or gated-attention pooling.

    Positions whose 3x3 neighbourhood touches the background are left out, so the leaf
    margin cannot drive the prediction; a leaf too small to keep any falls back to its full mask.

    pooling="mean": every leaf position gets the same weight.
    pooling="attention": gated attention (Ilse et al. 2018, "Attention-based Deep Multiple
    Instance Learning", ICML) learns one weight per position, so a few symptomatic patches
    can decide the leaf.
    """

    def __init__(self, channels: int, pooling: str = "mean", hidden: int = 32):
        super().__init__()
        assert pooling in ("mean", "attention")
        self.pooling = pooling
        self.out_dim = channels   # length of the pooled vector
        if pooling == "attention":
            self.att_v = nn.Linear(channels, hidden)
            self.att_u = nn.Linear(channels, hidden)
            self.att_w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool each leaf's positions.

        B: batch size. C: channels. H, W: map size.

        :param h: Features, shape (B, C, H, W).
        :type h: torch.Tensor
        :param mask: 1 on leaf positions, shape (B, 1, H, W).
        :type mask: torch.Tensor
        :return: pooled vectors (B, out_dim) and the weight of each position (B, H, W), summing to 1 per leaf.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
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
        z = (a.unsqueeze(-1) * feats).sum(1)
        return z, a.view(b, hh, ww)
