"""Masked layers and canvas helpers shared by the whole-leaf CNNs."""
from __future__ import annotations

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


def pad_to_canvas(cubes: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Centre every leaf on one square canvas; background and padding are 0.

    The canvas side is the largest leaf height or width, rounded up to a multiple
    of 4 (two 2x2 pools). Square so a 90° rotation keeps the leaf on the canvas.
    Offsets are multiples of 4 too, so the pooling grid lines up the same way
    for every leaf.

    n: the number of leaves. S: canvas side.

    :param cubes: n cubes, each (204, lines, samples) float32, background 0.
    :type cubes: list[np.ndarray]
    :param device: Where to keep the canvas.
    :type device: torch.device
    :return: X (n, 204, S, S) float16 reflectance, mask (n, 1, S, S) float32, S.
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    side = max(max(c.shape[1], c.shape[2]) for c in cubes)
    side = -(-side // 4) * 4
    X = torch.zeros(len(cubes), cubes[0].shape[0], side, side, dtype=torch.float16, device=device)
    for i, c in enumerate(cubes):
        _, h, w = c.shape
        top, left = ((side - h) // 2) // 4 * 4, ((side - w) // 2) // 4 * 4
        X[i, :, top:top + h, left:left + w] = torch.from_numpy(c).to(device, torch.float16)
    mask = (X != 0).any(dim=1, keepdim=True).float()
    return X, mask, side


def band_stats(X: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and SD of each band over the leaf pixels of the given (training) leaves.

    :param X: Canvas, shape (n, 204, S, S).
    :type X: torch.Tensor
    :param mask: Leaf mask, shape (n, 1, S, S).
    :type mask: torch.Tensor
    :param idx: Indices of the leaves to use, length k.
    :type idx: torch.Tensor
    :return: mean (204,) and SD (204,).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    s1 = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    s2, n = torch.zeros_like(s1), torch.zeros((), dtype=torch.float64, device=X.device)
    for i in idx.tolist():                 # one leaf at a time keeps memory small
        x, m = X[i].double(), mask[i].double()
        s1 += (x * m).sum((1, 2))
        s2 += (x * x * m).sum((1, 2))
        n += m.sum()
    mean = s1 / n
    return mean.float(), torch.sqrt(s2 / n - mean ** 2).float()


def augment(x: torch.Tensor, m: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random 90° rotations and flips, the same for a leaf's cube and mask.

    Rotation by 90° maps the square canvas onto itself. A moved leaf can sit 1-3 px off
    the pooling grid (training only; test leaves stay on it), which adds a little variety.

    :param x: Input, shape (B, 204, S, S).
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
