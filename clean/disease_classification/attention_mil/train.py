"""Canvas padding, preprocessing, augmentation and training for the whole-leaf CNN."""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from cnn_common import augment, band_stats, pad_to_canvas  # noqa: E402, F401
from data import load_capture_ids  # noqa: E402
from pca_common import snv  # noqa: E402
from model import LeafCNN  # noqa: E402

ROWS = ["R1", "R2", "R3", "R4", "R5"]
TASKS = {"leafroll": "leafroll3", "red_blotch": "red_blotch"}
BRIGHTNESS = ("band_z", "snv")
POOLING = ("mean", "attention")


@dataclass
class TrainConfig:
    """Training settings shared by every run (none were tuned on the test rows)."""

    batch: int = 8             # leaves per step
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 80
    patience: int = 15         # epochs without a lower validation loss
    brightness_jitter: float = 0.2   # band_z only: leaf brightness × U(0.8, 1.2); 0 turns it off
    dropout: float = 0.5             # before the last linear layer


def preprocess(x: torch.Tensor, m: torch.Tensor, brightness: str, stats, train: bool,
               cfg: TrainConfig, g: torch.Generator) -> torch.Tensor:
    """Turn a batch of reflectance canvases into network input; background stays 0.

    band_z: random leaf brightness (training only, unless cfg.brightness_jitter is 0),
            then each band standardised with training-leaf statistics. The network
            still sees absolute brightness.
    snv:    each pixel's spectrum centred and scaled over its own 204 bands (SNV),
            which removes brightness (light, leaf angle) and keeps spectral shape.
            No jitter: SNV would undo a brightness scale anyway.

    B: batch size. S: canvas side.

    :param x: Reflectance, shape (B, 204, S, S).
    :type x: torch.Tensor
    :param m: Leaf mask, shape (B, 1, S, S).
    :type m: torch.Tensor
    :param brightness: "band_z" or "snv".
    :type brightness: str
    :param stats: (mean, SD) per band from :func:`band_stats`; used by band_z.
    :type stats: tuple
    :param train: Apply the brightness jitter (band_z only).
    :type train: bool
    :param cfg: Brightness jitter size.
    :type cfg: TrainConfig
    :param g: Seeded generator for the jitter.
    :type g: torch.Generator
    :return: Input, shape (B, 204, S, S) float32.
    :rtype: torch.Tensor
    """
    x = x.float()
    if train and brightness == "band_z" and cfg.brightness_jitter > 0:
        lo, hi = 1 - cfg.brightness_jitter, 1 + cfg.brightness_jitter
        x = x * (lo + (hi - lo) * torch.rand(x.shape[0], 1, 1, 1, generator=g, device=x.device))
    if brightness == "band_z":
        mean, sd = stats
        x = (x - mean[None, :, None, None]) / sd[None, :, None, None]
    else:
        # SNV per pixel; background pixels have SD 0, so clamp and re-mask afterwards.
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = (x - mu) / sd
    return x * m


# How leaves are split into 5 folds. Every scheme keeps a photo (the camera image the leaves
# were cut from) inside one fold, so its shared lighting never spans training and test.
FOLD_SCHEMES = {
    "row": "fold k = field row k: the test row is never seen in training (new part of the vineyard)",
    "photo": "random photos dealt evenly per row (2 per fold; R1 has 13 photos, so 3 or 2): "
             "new vines in rows seen in training",
    "block": "fold k = the photos of each row whose first vine is in 10k+1 to 10k+10: like photo, but the "
             "held-out vines are neighbours of each other, not of the training vines (except at block edges)",
}
FOLD_SEED = 0  # fixes the random photo assignment of the "photo" scheme


def fold_of(meta, scheme: str) -> np.ndarray:
    """Give every leaf its fold 0-4 under one fold scheme.

    Photo = the camera image a leaf was cut from, from :func:`data.load_capture_ids`.
    Rows R2-R5 have 10 photos of 5 leaves; R1 has 12 of 4 and one of 2.

    :param meta: Leaf table with row and vine, n rows.
    :type meta: pandas.DataFrame
    :param scheme: "row", "photo" or "block" (see FOLD_SCHEMES).
    :type scheme: str
    :return: Fold per leaf, length n.
    :rtype: np.ndarray
    """
    rows = meta.row.to_numpy()
    if scheme == "row":
        return np.array([ROWS.index(r) for r in rows])
    photo = load_capture_ids(meta.path)
    if scheme == "block":
        # A photo goes where its first vine falls, so about 10 consecutive vines per fold.
        first_vine = meta.vine.groupby(photo).transform("min").to_numpy()
        return (first_vine - 1) // 10
    if scheme == "photo":
        # Inside each row, shuffle the photos and deal them out in turn, so every fold
        # holds about the same number of photos from every row.
        g = torch.Generator().manual_seed(FOLD_SEED)
        fold = np.zeros(len(meta), dtype=int)
        for r in ROWS:
            row_photos = np.unique(photo[rows == r])
            order = torch.randperm(len(row_photos), generator=g).numpy()
            for pos, i in enumerate(order):
                fold[photo == row_photos[i]] = pos % 5
        return fold
    raise ValueError(f"unknown fold scheme {scheme!r}; choose from {list(FOLD_SCHEMES)}")


def fold_split(fold: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = fold k, validation (early stopping) = fold k+1, train = the other 3 folds.

    :param fold: Fold per leaf from :func:`fold_of`, length n.
    :type fold: np.ndarray
    :param k: Fold number 0-4.
    :type k: int
    :return: train, validation, test leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(fold))
    val = (k + 1) % 5
    return idx[(fold != k) & (fold != val)], idx[fold == val], idx[fold == k]


def _predict(model, X, mask, idx, brightness, stats, cfg, g, batch: int = 16):
    """Logits and pooling weights for the given leaves, no augmentation (eval mode)."""
    logits, att = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            x = preprocess(X[j], mask[j], brightness, stats, False, cfg, g)
            lo, a = model(x, mask[j])
            logits.append(lo)
            att.append(a)
    return torch.cat(logits), torch.cat(att)


def train_fold(X: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, split, brightness: str,
               pooling: str, seed: int, cfg: TrainConfig) -> dict:
    """Train one CNN on one fold split and score its validation and test leaves.

    Early stopping uses the validation row's loss (the same class-weighted binary
    cross-entropy as training): 50 leaves give a noisy AUC but a steadier loss.
    The weights from the epoch with the lowest validation loss are restored.

    n: leaves of the variety. S4: side of the last feature map.

    :param X: Reflectance canvas, shape (n, 204, S, S) float16.
    :type X: torch.Tensor
    :param mask: Leaf mask, shape (n, 1, S, S).
    :type mask: torch.Tensor
    :param y: 0/1 label per leaf, shape (n,) float32.
    :type y: torch.Tensor
    :param split: (train, validation, test) index arrays from :func:`fold_split`.
    :type split: tuple
    :param brightness: "band_z" or "snv".
    :type brightness: str
    :param pooling: "mean" or "attention".
    :type pooling: str
    :param seed: Seed for initial weights, batch order and augmentation.
    :type seed: int
    :param cfg: Training settings.
    :type cfg: TrainConfig
    :return: test probabilities (n_test,), test pooling weights (n_test, S4, S4),
        best epoch, epochs run, per-epoch train and validation losses.
    :rtype: dict
    """
    dev = X.device
    tr, va, te = (torch.as_tensor(s, device=dev) for s in split)
    # Seed everything this run draws from: weights (global seed), batches and jitter (g).
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    stats = band_stats(X, mask, tr) if brightness == "band_z" else None

    model = LeafCNN(X.shape[1], pooling, cfg.dropout).to(dev)
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
            x = preprocess(X[j], mask[j], brightness, stats, True, cfg, g)
            x, m = augment(x, mask[j], g)
            logit, _ = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y[j], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        hist_tr.append(total / len(tr))

        model.eval()
        lv, _ = _predict(model, X, mask, va, brightness, stats, cfg, g)
        vloss = F.binary_cross_entropy_with_logits(lv, y[va], pos_weight=pos_weight).item()
        hist_va.append(vloss)
        if vloss < best[0]:
            best = (vloss, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= cfg.patience:
            break

    model.load_state_dict(best[2])
    model.eval()
    lt, att = _predict(model, X, mask, te, brightness, stats, cfg, g)
    return {
        "prob": torch.sigmoid(lt),
        "attention": att,
        "best_epoch": best[1],
        "epochs": len(hist_tr),
        "train_loss": hist_tr,
        "val_loss": hist_va,
    }


def baseline_fold(means: torch.Tensor, y: np.ndarray, train_idx: np.ndarray,
                  test_idx: np.ndarray) -> np.ndarray:
    """Logistic regression on the SNV leaf-mean spectrum (the earlier baseline).

    Why scikit-learn: it is the model the archived per-variety baseline used
    (StandardScaler + LogisticRegression, class_weight="balanced").

    :param means: Leaf-mean reflectance, shape (n, 204).
    :type means: torch.Tensor
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param train_idx: Training leaves.
    :type train_idx: np.ndarray
    :param test_idx: Test leaves.
    :type test_idx: np.ndarray
    :return: P(positive) for the test leaves, length n_test.
    :rtype: np.ndarray
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Z = snv(means.double()).cpu().numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(Z[train_idx], y[train_idx])
    return clf.predict_proba(Z[test_idx])[:, 1]
