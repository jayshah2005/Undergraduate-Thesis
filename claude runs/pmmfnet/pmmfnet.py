"""PMMFNet - Progressive Multi-scale Multi-attention Fusion network.

Re-implementation of
  H. Wang, S. Quan, J. Liu, H. Xiao, Y. Peng, Z. Wang, H. Li,
  "Progressive multi-scale multi-attention fusion for hyperspectral image
  classification", Scientific Reports 15:29288 (2025).
No official code was published; everything here is read off the text and
Figs. 3-8 of the paper. Where text and figures disagree, the TEXT is followed.
Choices the paper leaves open are marked [unspecified].

INPUT  (B, 1, 25, n_bands): one 5x5 neighbourhood flattened to 25 rows (row-major,
       centre pixel = row 12) x the bands, zero-filled to n_bands. This 2-D layout
       comes from the same group's spatial-shuffle paper (ref. 48: Wang, Cao & Liu,
       Remote Sens. 15:3960, 2023): "A sample with an N x N neighborhood and B
       bands is flattened into an image with B width and N x N [height]".
       The paper zero-fills to 200 bands; n_bands must be a multiple of 8 so the
       three 1x2 pools divide evenly (204 bands -> 208).
OUTPUT (B, n_out) logits.

DATA FLOW (Fig. 3; per-branch detail Figs. 6, 7, 8). 1x3 convs and 1x2 pools act
along the band axis only, so the 25 pixel rows are kept and the band axis
halves at each SWCA: 208 -> 104 -> 52 -> 26 (paper: 200 -> ... -> 25). All three
branches therefore meet at the same 25 x 26 map (paper: M x M = 25 x 25), which
their concatenation needs.

  x 25x208 -> stem ("Conv+pool") ------------------------------------- s0 25x104
     P branch: s0 -> SWCA -> SWCA -> 6 convs (7,7,4,3,3,3) -> NAM        -> P 25x26
  s0 -> SWCA ----------------------------------------------------------- s1 25x52
     D branch: s1 -> SWCA -> 6 convs (5,5,5,5,4,3) -> GAM                -> D 25x26
  s1 -> SWCA ----------------------------------------------------------- s2 25x26
     I branch: s2 -> 6 x [conv (7,5,4,3,3,3) -> CBAM]                    -> I 25x26
  fusion (Fig. 5):  NAM([P,I])   CBAM([I,P,D])   GAM([D,I])
  flatten the three, concatenate, one fully connected layer -> logits

Reading of the paper, point by point:
* Tap points. The text says P takes "the output obtained after the initial SWCA
  processing" and D "the output from the CBAM branch after two SWCA extractions".
  Fig. 7 draws the trunk as conv -> pool -> attention (the stem) followed by two
  full SWCAs, i.e. the stem is an SWCA without the shortcut (a 1 -> C conv cannot
  have one). Counting the stem as the first SWCA puts P at s0 and D at s1, and is
  the only reading under which the band-pool counts in Figs. 6-8 (P: 2, D: 1,
  I: 3 including the stem) and the text's 1x3-conv counts (P: two, I: three,
  D: one) all end at the same map size.
* Branch-end attention (text over figures). The text says P "goes through a
  channel-spatial attention mechanism" and D gives "the output of channel and
  spatial attention operations" before being stacked; Figs. 6 and 8 draw none.
  The module is not named; each branch's own attention is used (NAM for P, GAM
  for D; the I branch already has CBAM after every conv) - an interpretation.
* Fusion (text over Fig. 3): "flattened and concatenated, and then transformed
  ... through the fully connected layer" (Fig. 5 agrees); Fig. 3 draws ADD, which
  would also be impossible with 2C, 3C and 2C channels.
* SWCA shortcut: the text first says "concatenate" and then "each element ... is
  added to the corresponding element"; the specific sentence and Fig. 4's sum
  sign agree, so it is a sum.
* "both the detailed branch and the context branch consisting of 3 layers,
  without BN and ReLU" is in the PID-controller analogy paragraph (a toy
  controller), not the PMMFNet description; the SWCA text specifies "batch
  normalization and activation", so convs are Conv-BN-ReLU throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(c_in: int, c_out: int, k) -> nn.Sequential:
    """'conv+bn+relu' block of Figs. 6-8. 'same' padding keeps the map size,
    including for the even 4x4 kernels (PyTorch pads those asymmetrically)."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding="same", bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


# --------------------------------------------------------------------------
# SWCA - Shortcut Weight Channel Attention (Fig. 4, eqs. 2-5)
# --------------------------------------------------------------------------
class SWCA(nn.Module):
    """1x3 Conv-BN-ReLU + shortcut sum -> 1x2 max-pool -> channel weights.

    The channel weights are ECA-style: global average pooling -> normalisation
    with batch statistics and learnable alpha, beta (eqs. 2-4, i.e. BatchNorm1d)
    -> 1-D convolution over the channel axis with kernel k = 5 (Fig. 4) ->
    sigmoid (eq. 5) -> multiply the pooled map. Figs. 6-8 draw one 1x3 conv per
    SWCA (the text counts "two 1x3 convolutions" for the two SWCAs of branch P),
    so the Conv-BN-ReLU drawn after the multiply in Fig. 4 is the next layer,
    not a second conv inside the module.
    shortcut=False gives the stem (conv -> pool -> attention, Fig. 7).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 5, shortcut: bool = True):
        super().__init__()
        assert not shortcut or c_in == c_out, "the shortcut sum needs c_in == c_out"
        self.shortcut = shortcut
        self.conv = conv_bn_relu(c_in, c_out, (1, 3))
        self.pool = nn.MaxPool2d((1, 2))
        self.norm = nn.BatchNorm1d(c_out)                                   # eqs. (2)-(4)
        self.conv1d = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)        # eq. (5), C1D_k

    def forward(self, x):
        y = self.conv(x)
        f = self.pool(x + y if self.shortcut else y)
        s = self.norm(f.mean(dim=(2, 3)))                                   # GAP -> (B, C)
        w = torch.sigmoid(self.conv1d(s.unsqueeze(1))).squeeze(1)           # omega, (B, C)
        return f * w[:, :, None, None]


# --------------------------------------------------------------------------
# The three attention modules
# --------------------------------------------------------------------------
class NAM(nn.Module):
    """Normalization-based Attention Module (Liu et al. 2021; paper Fig. 6a-b, eq. 6).

    Channel sub-module: sigmoid(W_gamma * BN(F1)) with W_gamma = |gamma_i| / sum|gamma_j|.
    Spatial sub-module: the same with "pixel normalisation" - batch norm whose
    scale factors lambda are per spatial position (hw = H*W positions).
    Channel first, then spatial (F1 -> F2). As in the official NAM code, the
    weight normalisation uses the scale factors without gradient (.detach()).
    The NAM paper's optional L1 penalty on the scale factors is not used
    (PMMFNet does not mention it).
    """

    def __init__(self, c: int, hw: int):
        super().__init__()
        self.bn_c = nn.BatchNorm2d(c)
        self.bn_s = nn.BatchNorm1d(hw)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.bn_c.weight.detach().abs()
        x = x * torch.sigmoid(self.bn_c(x) * (g / g.sum()).view(1, -1, 1, 1))
        p = x.flatten(2).transpose(1, 2)                                     # (B, HW, C): positions as BN channels
        lam = self.bn_s.weight.detach().abs()
        s = self.bn_s(p) * (lam / lam.sum()).view(1, -1, 1)
        return x * torch.sigmoid(s.transpose(1, 2).reshape(b, c, h, w))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018; paper Fig. 7, eqs. 7-8).

    Channel: sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F))), shared MLP with
    reduction r = 16 (CBAM default) [unspecified]. Spatial: sigmoid(f7x7([avg; max]))
    over the channel axis.
    """

    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        h = max(c // r, 1)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        a = self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3)))     # eq. (7)
        x = x * torch.sigmoid(a)[:, :, None, None]
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.spatial(s))                            # eq. (8)


class GAM(nn.Module):
    """Global Attention Mechanism (Liu et al. 2021; paper Fig. 8a-c, eqs. 9-10).

    Channel: permute C x H x W -> H x W x C, two-layer MLP with ratio r, reverse
    permutation, sigmoid. Spatial: 7x7 conv (C -> C/r) -> BN -> ReLU -> 7x7 conv
    (C/r -> C) -> BN -> sigmoid, no pooling. r = 4 (GAM default) [unspecified].
    """

    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                     nn.Linear(c // r, c))
        self.spatial = nn.Sequential(
            nn.Conv2d(c, c // r, 7, padding=3), nn.BatchNorm2d(c // r), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 7, padding=3), nn.BatchNorm2d(c))

    def forward(self, x):
        a = self.channel(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x * torch.sigmoid(a)                                             # F2 = Mc(F1) * F1, eq. (9)
        return x * torch.sigmoid(self.spatial(x))                            # F3 = Ms(F2) * F2, eq. (10)


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
P_KERNELS = (7, 7, 4, 3, 3, 3)      # NAM (P) branch, Fig. 6 / text
I_KERNELS = (7, 5, 4, 3, 3, 3)      # CBAM (I) branch, Fig. 7 / text
D_KERNELS = (5, 5, 5, 5, 4, 3)      # GAM (D) branch, Fig. 8 / text


class PMMFNet(nn.Module):
    """channels: width of every conv layer [unspecified in the paper; 32 is what the
    same group's spatial-shuffle CNN (ref. 48) uses for its early layers]."""

    def __init__(self, n_out: int, channels: int = 32,
                 n_pixels: int = 25, n_bands: int = 208):
        super().__init__()
        assert n_bands % 8 == 0, "zero-fill the bands to a multiple of 8 (three 1x2 pools)"
        c, hw = channels, n_pixels * (n_bands // 8)

        self.stem = SWCA(1, c, shortcut=False)                               # s0: bands / 2
        self.trunk1 = SWCA(c, c)                                             # s1: bands / 4
        self.trunk2 = SWCA(c, c)                                             # s2: bands / 8

        self.p_branch = nn.Sequential(SWCA(c, c), SWCA(c, c),
                                      *[conv_bn_relu(c, c, k) for k in P_KERNELS], NAM(c, hw))
        self.i_branch = nn.Sequential(*[m for k in I_KERNELS
                                        for m in (conv_bn_relu(c, c, k), CBAM(c))])
        self.d_branch = nn.Sequential(SWCA(c, c),
                                      *[conv_bn_relu(c, c, k) for k in D_KERNELS], GAM(c))

        self.nam = NAM(2 * c, hw)
        self.cbam = CBAM(3 * c)
        self.gam = GAM(2 * c)
        self.fc = nn.Linear(7 * c * hw, n_out)

    def forward(self, x):                                                    # x: (B, 1, 25, n_bands)
        s0 = self.stem(x)
        s1 = self.trunk1(s0)
        s2 = self.trunk2(s1)
        p, i, d = self.p_branch(s0), self.i_branch(s2), self.d_branch(s1)
        f1 = self.nam(torch.cat([p, i], 1))                                  # Fig. 6: [1 | 2] -> NAM
        f2 = self.cbam(torch.cat([i, p, d], 1))                              # Fig. 7: [2 | 1 | 3] -> CBAM
        f3 = self.gam(torch.cat([d, i], 1))                                  # Fig. 8: [3 | 2] -> GAM
        return self.fc(torch.cat([f1.flatten(1), f2.flatten(1), f3.flatten(1)], 1))
