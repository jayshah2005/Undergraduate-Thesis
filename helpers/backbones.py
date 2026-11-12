"""Interchangeable CNN backbones for the whole-leaf models, chosen by name from BACKBONES.

Every backbone takes an SNV cube and its leaf mask and returns a feature map plus the
matching mask, 4× smaller on each side. The rules that keep a leaf's score independent of
its padding and position hold for all of them:
- normalisation uses leaf positions only (MaskedBatchNorm2d / 3d), and the background stays 0;
- the total stride is exactly 4, because pad_to_canvas aligns leaves to multiples of 4;
- the mask is shrunk with each downsampling, so a position stays leaf only if all of it is leaf.
The variety and disease models put a pooling layer and a linear head on top (see their model.py).

Each backbone follows a published design, adapted to whole leaves under those rules:
- small_cnn: this repo's own network (no paper); the attention pooling on top is Ilse et al. 2018.
- small_resnet: residual basic blocks, He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR.
- conv3d: HybridSN, Roy et al. 2020, "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral
  Image Classification", IEEE GRSL 17(2):277–281. A 3D CNN also beat a 2D CNN on Specim IQ grapevine
  leaves (virus detection) in Nguyen et al. 2021, Sensors 21(3):742.
- spectral_1d: the per-pixel 1D spectral CNN of Hu et al. 2015, "Deep Convolutional Neural Networks for
  Hyperspectral Image Classification", Journal of Sensors 2015:258619.
- resnet18_imagenet: ImageNet ResNet-18 (He et al. 2016) behind a linear band adaptor, the simple adaptor
  that worked best in Pérez & Maji 2021, "Domain Adaptor Networks for Hyperspectral Image Recognition",
  arXiv:2108.01555.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_common import MaskedBatchNorm2d, min_pool

DEFAULT_BACKBONE = "small_cnn"


class MaskedBatchNorm3d(MaskedBatchNorm2d):
    """MaskedBatchNorm2d for 3D feature maps (B, C, D, H, W): statistics over leaf positions and every depth."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalise x over leaf positions of every depth slice.

        B: batch size. C: channels. D: depth (spectral). H, W: map size.

        :param x: Features, shape (B, C, D, H, W).
        :type x: torch.Tensor
        :param mask: 1 on leaf positions, shape (B, 1, H, W).
        :type mask: torch.Tensor
        :return: Normalised features, same shape, 0 off the leaf.
        :rtype: torch.Tensor
        """
        b, c, d, h, w = x.shape
        # Each depth slice becomes its own "image", so the 2D layer's statistics cover all of them.
        flat = x.transpose(1, 2).reshape(b * d, c, h, w)
        y = super().forward(flat, mask.repeat_interleave(d, 0))
        return y.reshape(b, d, c, h, w).transpose(1, 2)


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


class SpectralSpatial3D(nn.Module):
    """HybridSN (Roy et al. 2020): three 3D convs over (band, y, x), then a 2D conv.

    The layers are HybridSN's: 3D kernels 7, 5 and 3 bands deep (3×3 in space) with 8, 16 and
    32 filters, then a 64-filter 3×3 2D conv. Adapted for whole leaves:
    - a learned 1×1 conv mixes the d bands into 24 (HybridSN uses PCA to 30, or 15). It keeps
      the 3D convs affordable and lets the same network take RGB;
    - the map is halved after the first 3D conv, because HybridSN runs on 25×25 patches, not
      212×212 leaves;
    - masked batch norm after each conv, and the shared pooling and head in place of its
      two fully connected layers.
    """

    depth = 24

    def __init__(self, n_bands: int):
        super().__init__()
        self.out_channels = 64
        self.spec = nn.Conv2d(n_bands, self.depth, 1, bias=False)
        self.spec_bn = MaskedBatchNorm2d(self.depth)
        self.c3 = nn.ModuleList([nn.Conv3d(1, 8, (7, 3, 3), padding=(0, 1, 1), bias=False),
                                 nn.Conv3d(8, 16, (5, 3, 3), padding=(0, 1, 1), bias=False),
                                 nn.Conv3d(16, 32, (3, 3, 3), padding=(0, 1, 1), bias=False)])
        self.bn3 = nn.ModuleList([MaskedBatchNorm3d(c) for c in (8, 16, 32)])
        d_out = self.depth - 6 - 4 - 2
        self.conv2d = nn.Conv2d(32 * d_out, 64, 3, padding=1, bias=False)
        self.bn2d = MaskedBatchNorm2d(64)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 64, S/4, S/4) and its mask."""
        h = F.relu(self.spec_bn(self.spec(x), mask)).unsqueeze(1)          # (B, 1, 24, S, S)
        h = F.relu(self.bn3[0](self.c3[0](h), mask))                        # (B, 8, 18, S, S)
        # Halve the map before the deeper 3D convs, which are the costly part.
        mask = min_pool(mask, 2, 2)
        h = F.max_pool3d(h, (1, 2, 2)) * mask[:, :, None]
        h = F.relu(self.bn3[1](self.c3[1](h), mask))                        # (B, 16, 14, S/2, S/2)
        h = F.relu(self.bn3[2](self.c3[2](h), mask))                        # (B, 32, 12, S/2, S/2)
        h = h.flatten(1, 2)                                                 # bands become channels
        h, mask = pool2(h, mask)
        return F.relu(self.bn2d(self.conv2d(h), mask)), mask


class Spectral1D(nn.Module):
    """Hu et al. 2015's spectral CNN on every leaf position: no spatial kernels at all.

    Each spectrum goes through Hu et al.'s layers: a 1D conv along the bands (20 filters of
    ⌈d/9⌉ bands), tanh, a 1D max-pool of ⌈k1/5⌉, and a fully connected layer of 100 units with
    tanh. Its 100 outputs are the position's features. Adapted for whole leaves: the cube is
    first averaged over 2×2 pixels (leaf windows only), which halves the cost, and the 100 features
    are max-pooled 2×2 once more so the output is S/4 like the other backbones. A score close to
    the 3×3 networks' says spatial pattern adds little.
    """

    n_filters, n_hidden = 20, 100

    def __init__(self, n_bands: int):
        super().__init__()
        self.out_channels = self.n_hidden
        k1 = math.ceil(n_bands / 9)
        self.k2 = math.ceil(k1 / 5)
        self.conv = nn.Conv1d(1, self.n_filters, k1)
        self.fc = nn.Linear(self.n_filters * ((n_bands - k1 + 1) // self.k2), self.n_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 100, S/4, S/4) and its mask."""
        mask = min_pool(mask, 2, 2)
        x = F.avg_pool2d(x, 2) * mask                                   # (B, d, S/2, S/2)
        b, d, h, w = x.shape
        spectra = x.permute(0, 2, 3, 1).reshape(-1, 1, d)                # one spectrum per position
        leaf = mask.flatten() > 0                                        # only leaf positions are scored
        z = F.max_pool1d(torch.tanh(self.conv(spectra[leaf])), self.k2)
        feats = torch.zeros(b * h * w, self.n_hidden, device=x.device, dtype=z.dtype)
        feats[leaf] = torch.tanh(self.fc(z.flatten(1)))
        return pool2(feats.view(b, h, w, -1).permute(0, 3, 1, 2), mask)


class ImageNetResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 (He et al. 2016) up to layer2, behind a learned linear band adaptor (d → 3).

    The adaptor is the simple linear projection that Pérez & Maji 2021 found hard to beat
    when feeding hyperspectral images to an RGB network.

    Cut at layer2 and without the stem's max-pool, so the total stride is 4 like the others
    (the full network downsamples 32×, which a ~150 px leaf cannot spare and the padding
    rules cannot allow). Every BatchNorm becomes a MaskedBatchNorm2d that starts from the
    pretrained statistics. All weights are fine-tuned.
    """

    def __init__(self, n_bands: int, pretrained: bool = True):
        super().__init__()
        self.out_channels = 128
        self.adapter = nn.Conv2d(n_bands, 3, 1, bias=False)
        self.adapter_bn = MaskedBatchNorm2d(3)   # ImageNet inputs are standardised per channel too
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = MaskedBatchNorm2d(64)
        self.layer1 = nn.ModuleList([MaskedBasicBlock(64, 64), MaskedBasicBlock(64, 64)])
        self.layer2 = nn.ModuleList([MaskedBasicBlock(64, 128, 2), MaskedBasicBlock(128, 128)])
        if pretrained:
            self._load_imagenet()

    def _load_imagenet(self) -> None:
        """Copy torchvision's ImageNet weights into the stem, layer1 and layer2."""
        from torchvision.models import ResNet18_Weights, resnet18

        src = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        def copy_bn(dst: MaskedBatchNorm2d, bn: nn.BatchNorm2d) -> None:
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(dst, name).data.copy_(getattr(bn, name).data)

        with torch.no_grad():
            self.conv1.weight.copy_(src.conv1.weight)
            copy_bn(self.bn1, src.bn1)
            for mine, theirs in ((self.layer1, src.layer1), (self.layer2, src.layer2)):
                for a, b in zip(mine, theirs):
                    a.conv1.weight.copy_(b.conv1.weight)
                    a.conv2.weight.copy_(b.conv2.weight)
                    copy_bn(a.bn1, b.bn1)
                    copy_bn(a.bn2, b.bn2)
                    if a.down is not None:
                        a.down.weight.copy_(b.downsample[0].weight)
                        copy_bn(a.down_bn, b.downsample[1])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature map (B, 128, S/4, S/4) and its mask."""
        h = self.adapter_bn(self.adapter(x), mask)
        mask = min_pool(mask, 2, 2)
        h = F.relu(self.bn1(self.conv1(h), mask))
        for block in (*self.layer1, *self.layer2):
            h, mask = block(h, mask)
        return h, mask


BACKBONES = {
    "small_cnn": SmallCNN,
    "small_resnet": SmallResNet,
    "conv3d": SpectralSpatial3D,
    "spectral_1d": Spectral1D,
    "resnet18_imagenet": ImageNetResNet18,
}
# How each backbone is named in the reports: "CNN" for the original keeps the older report names.
LABEL = {
    "small_cnn": "CNN",
    "small_resnet": "small ResNet",
    "conv3d": "3D CNN",
    "spectral_1d": "1D spectral CNN",
    "resnet18_imagenet": "ImageNet ResNet-18",
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
