"""Interchangeable CNN backbones for the whole-leaf models, chosen by name from BACKBONES.

Every backbone takes an SNV cube and its leaf mask and returns a feature map plus the
matching mask, 4× smaller on each side. The rules that keep a leaf's score independent of
its padding and position hold for all of them:
- normalisation uses leaf positions only (MaskedBatchNorm2d), and the background stays 0;
- the total stride is exactly 4, because pad_to_canvas aligns leaves to multiples of 4;
- the mask is shrunk with each downsampling, so a position stays leaf only if all of it is leaf.
The variety and disease models put a pooling layer and a linear head on top (see their model.py).

Each backbone follows a published design, adapted to whole leaves under those rules:
- small_cnn: this repo's own network (no paper); the attention pooling on top is Ilse et al. 2018.
- small_resnet: residual basic blocks, He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR.
Three more were screened and removed (HybridSN 3D CNN, Hu et al. 2015 1D spectral CNN, ImageNet ResNet-18 with a
band adaptor); they are in git history at fb310b9.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_common import MaskedBatchNorm2d, min_pool

DEFAULT_BACKBONE = "small_cnn"


def pool2(h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2×2 max pool of the features; a pooled position is leaf only if all four under it are leaf."""
    mask = min_pool(mask, 2, 2)
    return F.max_pool2d(h, 2) * mask, mask


class SmallCNN(nn.Module):
    """The original network: 1×1 spectral conv → 4 masked 3×3 conv layers with 2 max-pools.

    This repo's own design, not from a paper. Every position of the output stands for a
    ~26 px patch of leaf (the receptive field).
    """

    widths = (16, 24, 48)   # 1x1 spectral conv, block 1, blocks 2-3

    def __init__(self, n_bands: int):
        super().__init__()
        c0, c1, c2 = self.widths
        self.out_channels = c2
        self.spec = nn.Conv2d(n_bands, c0, 1, bias=False)   # learned band reduction
        self.conv1 = nn.Conv2d(c0, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn = nn.ModuleList([MaskedBatchNorm2d(c) for c in (c0, c1, c2, c2, c2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 48, S/4, S/4) and its mask (B, 1, S/4, S/4) from x (B, d, S, S) and mask (B, 1, S, S)."""
        h = F.relu(self.bn[0](self.spec(x), mask))
        h = F.relu(self.bn[1](self.conv1(h), mask))
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = F.relu(self.bn[2](self.conv2(h * mask), mask))
        h, mask = F.max_pool2d(h, 2), min_pool(mask, 2, 2)
        h = F.relu(self.bn[3](self.conv3(h * mask), mask))
        h = F.relu(self.bn[4](self.conv4(h), mask))
        return h, mask


class MaskedBasicBlock(nn.Module):
    """ResNet basic block (He et al. 2016: two 3×3 convs plus a skip connection) with masked batch norm.

    Attribute names follow torchvision's BasicBlock, so pretrained ResNet-18 weights load into it.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = MaskedBatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = MaskedBatchNorm2d(c_out)
        # A 1×1 conv on the skip path when the shape changes, as in torchvision.
        self.down = (nn.Conv2d(c_in, c_out, 1, stride, bias=False) if stride != 1 or c_in != c_out else None)
        self.down_bn = MaskedBatchNorm2d(c_out) if self.down is not None else None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the block; a stride-2 block also halves the mask."""
        m = min_pool(mask, 2, 2) if self.stride == 2 else mask
        out = F.relu(self.bn1(self.conv1(h), m))
        out = self.bn2(self.conv2(out), m)
        skip = h if self.down is None else self.down_bn(self.down(h), m)
        return F.relu(out + skip), m


class SmallResNet(nn.Module):
    """1×1 spectral conv → ResNet basic blocks (He et al. 2016) of 32, 64, 64 channels with 2 max-pools.

    Four blocks (8 conv layers) instead of ResNet-18's eight, because 1000 leaves cannot train a full one.
    """

    def __init__(self, n_bands: int):
        super().__init__()
        self.out_channels = 64
        self.spec = nn.Conv2d(n_bands, 32, 1, bias=False)
        self.spec_bn = MaskedBatchNorm2d(32)
        self.stage1 = nn.ModuleList([MaskedBasicBlock(32, 32)])
        self.stage2 = nn.ModuleList([MaskedBasicBlock(32, 64)])
        self.stage3 = nn.ModuleList([MaskedBasicBlock(64, 64), MaskedBasicBlock(64, 64)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 64, S/4, S/4) and its mask."""
        h = F.relu(self.spec_bn(self.spec(x), mask))
        for i, stage in enumerate((self.stage1, self.stage2, self.stage3)):
            if i:
                h, mask = pool2(h, mask)
            for block in stage:
                h, mask = block(h, mask)
        return h, mask


BACKBONES = {
    "small_cnn": SmallCNN,
    "small_resnet": SmallResNet,
}
# How each backbone is named in the reports: "CNN" for the original keeps the older report names.
LABEL = {
    "small_cnn": "CNN",
    "small_resnet": "small ResNet",
}


def make_backbone(name: str, n_bands: int) -> nn.Module:
    """Build a backbone from BACKBONES by name.

    :param name: A key of BACKBONES.
    :type name: str
    :param n_bands: Input bands (204, or 3 for RGB).
    :type n_bands: int
    :return: A module mapping (x (B, d, S, S), mask (B, 1, S, S)) to (features (B, C, S/4, S/4), mask);
        C is its out_channels.
    :rtype: nn.Module
    """
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name](n_bands)


def masked_norms(model: nn.Module) -> list[MaskedBatchNorm2d]:
    """Every masked batch-norm layer in a model (the invariance checks give them non-trivial statistics)."""
    return [m for m in model.modules() if isinstance(m, MaskedBatchNorm2d)]
