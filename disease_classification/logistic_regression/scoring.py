"""Disease metrics and their bootstrap, shared by the baseline, the CNN's training log and the CNN report."""
from __future__ import annotations

import numpy as np
import torch

N_BOOT, BOOT_SEED = 2000, 0
METRICS = ("bal_acc", "accuracy", "sensitivity", "specificity", "precision", "f1")  # metrics.md columns
THRESHOLD = 0.5  # a leaf is called positive when P(positive) >= 0.5; fixed in advance, never tuned


def balanced_accuracy(p: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of the recall on positive leaves and the recall on negative leaves, at THRESHOLD.

    With bootstrap weights w (B, n), returns one balanced accuracy per resample.

    n: the number of leaves. B: the number of resamples.

    :param p: P(positive), shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: Balanced accuracy, shape () or (B,).
    :rtype: torch.Tensor
    """
    pos = (y == 1).double()
    hit = ((p >= THRESHOLD).double() == pos).double()
    if w is None:
        w = torch.ones_like(pos)
    recall_pos = (w * hit * pos).sum(-1) / (w * pos).sum(-1)
    recall_neg = (w * hit * (1 - pos)).sum(-1) / (w * (1 - pos)).sum(-1)
    return (recall_pos + recall_neg) / 2


def binary_metrics(p: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> dict:
    """Every yes/no metric at THRESHOLD, from (weighted) confusion counts.

    Sensitivity (= recall) is the recall on positive leaves, specificity the recall on negative
    leaves; balanced accuracy is their mean, the same value as :func:`balanced_accuracy`.
    Precision is undefined (NaN) when no leaf is called positive.

    n: the number of leaves. B: the number of resamples.

    :param p: P(positive), shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: {metric: tensor of shape () or (B,)}: bal_acc, accuracy, sensitivity, specificity,
        precision, f1, and the counts tp, fn, fp, tn.
    :rtype: dict
    """
    pos = (y == 1).double()
    call = (p >= THRESHOLD).double()
    if w is None:
        w = torch.ones_like(pos)
    tp, fn = (w * call * pos).sum(-1), (w * (1 - call) * pos).sum(-1)
    fp, tn = (w * call * (1 - pos)).sum(-1), (w * (1 - call) * (1 - pos)).sum(-1)
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"bal_acc": (sens + spec) / 2, "accuracy": (tp + tn) / (tp + fn + fp + tn), "sensitivity": sens,
            "specificity": spec, "precision": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def all_metrics(p: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> dict:
    """Every metric of :func:`binary_metrics` with its 95% interval over the resamples w.

    n: the number of leaves. B: the number of resamples.

    :param p: P(positive), shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param w: Resampling counts, shape (B, n).
    :type w: torch.Tensor
    :return: {metric: value} plus {metric + "_ci95": [lower, upper]} for each rate, and the counts.
    :rtype: dict
    """
    point, boot = binary_metrics(p, y), binary_metrics(p, y, w)
    q = torch.tensor([0.025, 0.975], dtype=torch.float64, device=p.device)
    out = {k: v.item() for k, v in point.items()}
    for k in METRICS:
        out[k + "_ci95"] = torch.nanquantile(boot[k], q).tolist()   # nan: precision with no positive call
    return out


def bootstrap_weights(y: torch.Tensor, group: torch.Tensor, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> torch.Tensor:
    """Resample leaves with replacement within each (group, class) cell.

    Keeping each cell's size fixed keeps both recalls defined in every resample and keeps
    each row's class balance. Seeded so every model sees the same resamples, which makes
    the model-minus-baseline differences paired.

    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param group: Group per leaf (variety × row), shape (n,).
    :type group: torch.Tensor
    :param n_boot: Number of resamples B.
    :type n_boot: int
    :param seed: Generator seed.
    :type seed: int
    :return: Counts, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    g = torch.Generator(device=y.device).manual_seed(seed)
    w = torch.zeros(n_boot, len(y), dtype=torch.float64, device=y.device)
    for k in torch.unique(group).tolist():
        for c in (0, 1):
            idx = torch.nonzero((group == k) & (y == c)).squeeze(1)
            if len(idx) == 0:
                continue
            draw = idx[torch.randint(len(idx), (n_boot, len(idx)), generator=g, device=y.device)]
            w.scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
    return w


def summary(p_seeds: np.ndarray, y: np.ndarray, fold: np.ndarray, variety: np.ndarray,
            device: torch.device) -> dict:
    """Score one model on the leaves it predicted: its seed ensemble, each seed, each fold and each variety.

    The ensemble averages P(positive) over the seeds, then applies THRESHOLD.

    n: the leaves given. s: the number of seeds.

    :param p_seeds: P(positive) per seed and leaf, shape (s, n).
    :type p_seeds: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param fold: Fold per leaf, length n.
    :type fold: np.ndarray
    :param variety: Variety per leaf, length n.
    :type variety: np.ndarray
    :param device: GPU or CPU.
    :type device: torch.device
    :return: bal_acc (ensemble), seeds (list), by_fold ({fold: value}) and by_variety
        ({variety: value}, empty for one variety).
    :rtype: dict
    """
    p = torch.tensor(p_seeds, dtype=torch.float64, device=device)
    labels = torch.tensor(y, dtype=torch.float64, device=device)
    ens = p.mean(0)

    def part(mask: np.ndarray) -> float:
        """Ensemble balanced accuracy over the leaves in mask."""
        m = torch.tensor(mask, device=device)
        return balanced_accuracy(ens[m], labels[m]).item()

    varieties = sorted(set(variety))
    return {"bal_acc": balanced_accuracy(ens, labels).item(),
            "seeds": [balanced_accuracy(s, labels).item() for s in p],
            "by_fold": {int(k): part(fold == k) for k in np.unique(fold)},
            "by_variety": {v: part(variety == v) for v in varieties} if len(varieties) > 1 else {}}


def per_variety_metrics(p: torch.Tensor, y: torch.Tensor, variety: np.ndarray) -> dict:
    """Point values of :func:`binary_metrics` for each variety; empty for one variety.

    n: the number of leaves.

    :param p: P(positive), shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param variety: Variety per leaf, length n.
    :type variety: np.ndarray
    :return: {variety: {metric: value}}.
    :rtype: dict
    """
    varieties = sorted(set(variety))
    if len(varieties) < 2:
        return {}
    out = {}
    for v in varieties:
        m = torch.tensor(variety == v, device=p.device)
        out[v] = {k: x.item() for k, x in binary_metrics(p[m], y[m]).items()}
    return out
