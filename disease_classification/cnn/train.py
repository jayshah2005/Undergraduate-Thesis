"""Preprocessing (SNV per pixel), augmentation and training for the whole-leaf CNN."""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.backbones import DEFAULT_BACKBONE  # noqa: E402
from helpers.cnn_common import augment  # noqa: E402
from model import LeafCNN  # noqa: E402

POOLING = ("mean", "attention")
# Canvas side for every disease model: the largest of all 998 leaves (211 px) rounded up to a
# multiple of 4, so every model, pooled or one variety, sees 212 × 212 × d inputs.
CANVAS = 212


@dataclass
class TrainConfig:
    """Training settings shared by every run (none were tuned on the test rows)."""

    batch: int = 8             # leaves per step
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 80
    patience: int = 15         # epochs without a lower validation loss
    dropout: float = 0.5       # before the last linear layer
    augment: bool = True       # random 90° rotations and flips while training
    backbone: str = DEFAULT_BACKBONE   # a key of helpers.backbones.BACKBONES; the same settings for every backbone


def preprocess(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply SNV to each pixel over its own d bands, as in the variety CNN; background stays 0.

    SNV centres and scales each pixel's spectrum, which removes brightness (light,
    leaf angle) and keeps spectral shape.

    B: batch size. d: input bands (204, or 3 for RGB). S: canvas side.

    :param x: Reflectance, shape (B, d, S, S).
    :type x: torch.Tensor
    :param m: Leaf mask, shape (B, 1, S, S).
    :type m: torch.Tensor
    :return: Input, shape (B, d, S, S) float32.
    :rtype: torch.Tensor
    """
    x = x.float()
    # Background pixels have SD 0, so clamp and re-mask afterwards.
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mu) / sd * m


def _predict(model, X, mask, idx, batch: int = 16):
    """Logits and pooling weights for the given leaves (eval mode)."""
    logits, att = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            x = preprocess(X[j], mask[j])
            lo, a = model(x, mask[j])
            logits.append(lo)
            att.append(a)
    return torch.cat(logits), torch.cat(att)


def train_fold(X: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, split, pooling: str, seed: int,
               cfg: TrainConfig) -> dict:
    """Train one CNN on one fold split and score its validation and test leaves.

    Early stopping uses the validation row's loss (the same class-weighted binary
    cross-entropy as training): 50 leaves give a noisy accuracy but a steadier loss.
    The weights from the epoch with the lowest validation loss are restored.

    n: leaves loaded. d: input bands. S4: side of the last feature map.

    :param X: Reflectance canvas, shape (n, d, S, S) float16.
    :type X: torch.Tensor
    :param mask: Leaf mask, shape (n, 1, S, S).
    :type mask: torch.Tensor
    :param y: 0/1 label per leaf, shape (n,) float32.
    :type y: torch.Tensor
    :param split: (train, validation, test) index arrays from logistic_regression.fold_split.
    :type split: tuple
    :param pooling: "mean" or "attention".
    :type pooling: str
    :param seed: Seed for initial weights, batch order and augmentation.
    :type seed: int
    :param cfg: Training settings.
    :type cfg: TrainConfig
    :return: test probabilities (n_test,), test pooling weights (n_test, S4, S4), validation
        probabilities (n_val,) from the kept weights, best epoch, epochs run, per-epoch train
        and validation losses.
    :rtype: dict
    """
    dev = X.device
    tr, va, te = (torch.as_tensor(s, device=dev) for s in split)
    # Seed everything this run draws from: weights (global seed), batch order and augmentation (g).
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)

    model = LeafCNN(X.shape[1], pooling, cfg.dropout, cfg.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_pos = y[tr].sum()
    # Up-weight the positive class by #neg / #pos so both classes count equally.
    pos_weight = (len(tr) - n_pos) / n_pos

    best = (float("inf"), 0, None)
    hist_tr, hist_va = [], []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g, device=dev)]
        total = 0.0
        for s in range(0, len(order), cfg.batch):
            j = order[s:s + cfg.batch]
            x, m = preprocess(X[j], mask[j]), mask[j]
            if cfg.augment:
                x, m = augment(x, m, g)
            logit, _ = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y[j], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        hist_tr.append(total / len(tr))

        model.eval()
        lv, _ = _predict(model, X, mask, va)
        vloss = F.binary_cross_entropy_with_logits(lv, y[va], pos_weight=pos_weight).item()
        hist_va.append(vloss)
        if vloss < best[0]:
            best = (vloss, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= cfg.patience:
            break

    model.load_state_dict(best[2])
    model.eval()
    lt, att = _predict(model, X, mask, te)
    lv, _ = _predict(model, X, mask, va)   # kept so backbones can be compared without the test rows
    return {
        "prob": torch.sigmoid(lt),
        "attention": att,
        "val_prob": torch.sigmoid(lv),
        "best_epoch": best[1],
        "epochs": len(hist_tr),
        "train_loss": hist_tr,
        "val_loss": hist_va,
    }
