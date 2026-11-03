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
from pca_common import snv  # noqa: E402
from model import LeafCNN  # noqa: E402

ROWS = ["R1", "R2", "R3", "R4", "R5"]
# What counts as positive in each task; every other leaf is negative.
TASKS = {
    "leafroll": "leafroll present",
    "red_blotch": "red blotch present",
    "both": "both diseases present; a leaf with only one counts as negative",
    "any": "any disease present; healthy leaves are the negatives",
}
MIN_PER_CLASS = 5  # a variety with fewer positive or negative leaves sits out that task
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


def fold_of(meta) -> np.ndarray:
    """Give every leaf its fold 0-4: fold k holds field row k of every variety.

    :param meta: Leaf table with row, n rows.
    :type meta: pandas.DataFrame
    :return: Fold per leaf, length n.
    :rtype: np.ndarray
    """
    return np.array([ROWS.index(r) for r in meta.row.to_numpy()])


def task_labels(meta, task: str) -> np.ndarray:
    """Give every leaf its 0/1 label for one task (see TASKS).

    :param meta: Leaf table with red_blotch and leafroll3, n rows.
    :type meta: pandas.DataFrame
    :param task: A key of TASKS.
    :type task: str
    :return: 0/1 label per leaf, length n.
    :rtype: np.ndarray
    """
    leafroll, red_blotch = meta.leafroll3.to_numpy() == 1, meta.red_blotch.to_numpy() == 1
    positive = {"leafroll": leafroll, "red_blotch": red_blotch,
                "both": leafroll & red_blotch, "any": leafroll | red_blotch}[task]
    return positive.astype(int)


def usable_leaves(meta, y: np.ndarray) -> np.ndarray:
    """Keep the varieties with at least MIN_PER_CLASS positive and negative leaves.

    Below that a variety can't be scored, and in a pooled model its leaves would only teach
    "this variety means this class". Cab Franc has 1 leaf with leafroll, for example.

    :param meta: Leaf table with variety, n rows.
    :type meta: pandas.DataFrame
    :param y: 0/1 label per leaf, length n.
    :type y: np.ndarray
    :return: True for the leaves this task uses, length n.
    :rtype: np.ndarray
    """
    varieties = meta.variety.to_numpy()
    keep = np.zeros(len(y), dtype=bool)
    for v in np.unique(varieties):
        in_variety = varieties == v
        if min(y[in_variety].sum(), (1 - y[in_variety]).sum()) >= MIN_PER_CLASS:
            keep |= in_variety
    return keep


def fold_split(fold: np.ndarray, k: int, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = row k, validation (early stopping) = row k+1, train = the other 3 rows.

    :param fold: Fold per leaf from :func:`fold_of`, length n.
    :type fold: np.ndarray
    :param k: Fold number 0-4.
    :type k: int
    :param keep: Leaves to split, from :func:`usable_leaves`; None means all of them.
    :type keep: np.ndarray | None
    :return: train, validation, test leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(fold)) if keep is None else np.flatnonzero(keep)
    f, val = fold[idx], (k + 1) % 5
    return idx[(f != k) & (f != val)], idx[f == val], idx[f == k]


def check_task(meta, fold: np.ndarray, task: str) -> None:
    """Stop before training if a task can't be run on these leaves and folds.

    :param meta: Leaf table with variety and labels, n rows.
    :type meta: pandas.DataFrame
    :param fold: Fold per leaf from :func:`fold_of`, length n.
    :type fold: np.ndarray
    :param task: A key of TASKS.
    :type task: str
    :raises ValueError: if no variety is usable, or a fold's train, validation or test set lacks a class.
    """
    y = task_labels(meta, task)
    keep = usable_leaves(meta, y)
    if not keep.any():
        raise ValueError(f"{task}: every variety here has fewer than {MIN_PER_CLASS} positive or negative leaves")
    for k in range(5):
        for name, part in zip(("train", "validation", "test"), fold_split(fold, k, keep)):
            if len(np.unique(y[part])) < 2:
                raise ValueError(f"{task}: the {name} set of test row {ROWS[k]} has only one class; "
                                 "try another variety")


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

    n: leaves loaded. S4: side of the last feature map.

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
