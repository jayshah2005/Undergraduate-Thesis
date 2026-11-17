"""Interchangeable CNN backbones for the whole-leaf models, chosen by name from BACKBONES.

Every backbone takes an SNV cube and its leaf mask and returns a feature map plus the
matching mask, `stride` times smaller on each side. The rules that keep a leaf's score
independent of its padding and position hold for all of them:
- normalisation uses leaf positions only (MaskedBatchNorm2d), and the background stays 0;
- leaves sit on the canvas at offsets that are multiples of the stride (pad_to_canvas(align=stride)),
  so every leaf lines up with the downsampling grid the same way;
- the mask is shrunk with each downsampling. In the stride-4 backbones a position stays leaf only
  if all of it is leaf; ResNet-18 keeps a position if any of it is leaf, because at stride 32 the
  all-leaf rule leaves 212 of the 998 leaves with no position at all.
The variety and disease models put a pooling layer and a linear head on top (see their model.py).

Each backbone follows a published design, adapted to whole leaves under those rules:
- small_cnn: this repo's own network (no paper); the attention pooling on top is Ilse et al. 2018.
- small_resnet: residual basic blocks, He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR.
- small_resnet_strided: small_resnet with its max pools replaced by stride-2 convs, the paper's own way of
  downsampling; Springenberg et al. 2015, "Striving for Simplicity: The All Convolutional Net", ICLR workshop,
  found max pooling can be replaced by a stride-2 conv without loss.
- resnet18: the 18-layer network of He et al. 2016 (their Table 1), with the fewest changes whole leaves need.
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
    stride = 4

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

    def __init__(self, c_in: int, c_out: int, stride: int = 1, any_leaf: bool = False):
        super().__init__()
        self.stride = stride
        self.any_leaf = any_leaf   # a stride-2 block keeps a position if any (True) or all (False) of it is leaf
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = MaskedBatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = MaskedBatchNorm2d(c_out)
        # A 1×1 conv on the skip path when the shape changes, as in torchvision.
        self.down = (nn.Conv2d(c_in, c_out, 1, stride, bias=False) if stride != 1 or c_in != c_out else None)
        self.down_bn = MaskedBatchNorm2d(c_out) if self.down is not None else None

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the block; a stride-2 block also halves the mask."""
        m = mask
        if self.stride == 2:
            m = F.max_pool2d(mask, 2, 2) if self.any_leaf else min_pool(mask, 2, 2)
        out = F.relu(self.bn1(self.conv1(h), m))
        out = self.bn2(self.conv2(out), m)
        skip = h if self.down is None else self.down_bn(self.down(h), m)
        return F.relu(out + skip), m


class SmallResNet(nn.Module):
    """1×1 spectral conv → ResNet basic blocks (He et al. 2016) of 32, 64, 64 channels with 2 max-pools.

    Four blocks (8 conv layers) instead of ResNet-18's eight, because 1000 leaves cannot train a full one.
    """

    stride = 4

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


class SmallResNetStrided(SmallResNet):
    """SmallResNet without max pooling: the first conv of stage2 and of stage3 has stride 2 instead.

    This is how He et al. 2016 downsample (conv3_1, conv4_1, conv5_1 have stride 2), and Springenberg
    et al. 2015 found a stride-2 conv can replace a max pool without loss. Both stride-2 blocks change
    the map's shape, so their skip gets a 1×1 conv + BN with stride 2 (option B). Everything else is
    SmallResNet's: widths, depth, total stride 4 and the all-leaf mask rule.
    """

    def __init__(self, n_bands: int):
        nn.Module.__init__(self)
        self.out_channels = 64
        self.spec = nn.Conv2d(n_bands, 32, 1, bias=False)
        self.spec_bn = MaskedBatchNorm2d(32)
        self.stage1 = nn.ModuleList([MaskedBasicBlock(32, 32)])
        self.stage2 = nn.ModuleList([MaskedBasicBlock(32, 64, stride=2)])
        self.stage3 = nn.ModuleList([MaskedBasicBlock(64, 64, stride=2), MaskedBasicBlock(64, 64)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 64, S/4, S/4) and its mask; the stride-2 blocks halve the map and the mask."""
        h = F.relu(self.spec_bn(self.spec(x), mask))
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                h, mask = block(h, mask)
        return h, mask


class ResNet18(nn.Module):
    """ResNet-18 as in He et al. 2016 (Table 1), trained from scratch, changed only where whole leaves need it.

    As in the paper: a 7×7 conv with 64 filters and stride 2, a 3×3 max pool with stride 2, then four
    stages of two basic blocks (64, 128, 256, 512 channels). Stages 3-5 (conv3_1, conv4_1, conv5_1 in
    the paper) halve the map with a stride-2 first conv, and their skip uses a 1×1 projection (option B).
    BN comes after every conv and before the ReLU; the ReLU after the addition. Convs start from He
    initialisation, as the paper and torchvision do. The total stride is 32: a 224 canvas ends as 7×7.

    Changed, and why:
    - the first conv reads the 204 bands (or 3 for RGB) instead of 3 colour channels;
    - every BatchNorm is a MaskedBatchNorm2d, so padding cannot change the statistics;
    - the mask shrinks by the any-leaf rule (see the module docstring), and leaves sit on a 32 px grid;
    - the average pool and the classifier are the shared leaf pooling and head of model.py, so they
      average over leaf positions, not over the padding.
    """

    stride = 32

    def __init__(self, n_bands: int):
        super().__init__()
        self.out_channels = 512
        self.conv1 = nn.Conv2d(n_bands, 64, 7, 2, 3, bias=False)
        self.bn1 = MaskedBatchNorm2d(64)
        layers, c_in = [], 64
        for c, s in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers.append(nn.ModuleList([MaskedBasicBlock(c_in, c, s, any_leaf=True),
                                         MaskedBasicBlock(c, c, any_leaf=True)]))
            c_in = c
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):   # He et al. 2015 initialisation, as in torchvision's ResNet
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 512, S/32, S/32) and its mask from x (B, d, S, S) and mask (B, 1, S, S)."""
        mask = F.max_pool2d(mask, 2, 2)
        h = F.relu(self.bn1(self.conv1(x), mask))
        mask = F.max_pool2d(mask, 2, 2)
        # Features are >= 0 after the ReLU and 0 off the leaf, so the max pool's -inf border padding
        # never wins over a real value.
        h = F.max_pool2d(h, 3, 2, 1) * mask
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                h, mask = block(h, mask)
        return h, mask


BACKBONES = {
    "small_cnn": SmallCNN,
    "small_resnet": SmallResNet,
    "small_resnet_strided": SmallResNetStrided,
    "resnet18": ResNet18,
}
# How each backbone is named in the reports: "CNN" for the original keeps the older report names.
LABEL = {
    "small_cnn": "CNN",
    "small_resnet": "small ResNet",
    "small_resnet_strided": "small ResNet (no max pool)",
    "resnet18": "ResNet-18",
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
