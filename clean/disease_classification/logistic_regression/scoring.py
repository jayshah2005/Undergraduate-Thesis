"""Balanced accuracy and its bootstrap, shared by the baseline, the CNN's training log and the CNN report."""
from __future__ import annotations

import numpy as np
import torch

N_BOOT, BOOT_SEED = 2000, 0
MACRO_TASKS = ("leafroll", "red_blotch")  # the macro average is the mean of these two balanced accuracies
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


def bootstrap_weights(y: torch.Tensor, group: torch.Tensor, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> torch.Tensor:
    """Resample leaves with replacement within each (group, class) cell.

    Keeping each cell's size fixed keeps both recalls defined in every resample and keeps
    each row's class balance. Seeded so every model sees the same resamples, which makes
    the model-minus-baseline differences paired.

    :param y: Class per leaf (0/1, or a joint code), shape (n,).
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
        for c in torch.unique(y).tolist():   # 0 and 1, or the joint codes of macro_weights
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


def macro_weights(labels: list[np.ndarray], group: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resamples shared by every task of the macro average, so its interval resamples leaves, not tasks.

    Each leaf gets one joint class (its label in every task, or "left out"), and leaves are
    resampled within each (group, joint class) cell, so every resample keeps each row's size
    and the rate of every disease combination.

    n: the number of leaves.

    :param labels: Per task, the 0/1 label per leaf or -1 where the task leaves the leaf out, length n each.
    :type labels: list[np.ndarray]
    :param group: Group per leaf (variety × row), length n.
    :type group: np.ndarray
    :param device: GPU or CPU.
    :type device: torch.device
    :return: Counts, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    code = np.zeros(len(group), dtype=np.int64)
    for y in labels:
        code = code * 3 + (y + 1)
    return bootstrap_weights(torch.tensor(code, device=device), torch.tensor(group, device=device))


def macro_score(p_seeds: list[np.ndarray], labels: list[np.ndarray], fold: np.ndarray, variety: np.ndarray,
                w: torch.Tensor, device: torch.device) -> dict:
    """Score one model by the mean of its balanced accuracies over the tasks.

    Each task's balanced accuracy uses only the leaves that task keeps; the macro value is
    their plain mean, so each task counts equally whatever its size.

    n: the number of leaves. s: the number of seeds. B: the number of resamples.

    :param p_seeds: Per task, P(positive) per seed and leaf, shape (s, n) each.
    :type p_seeds: list[np.ndarray]
    :param labels: Per task, the 0/1 label per leaf or -1 where the task leaves the leaf out.
    :type labels: list[np.ndarray]
    :param fold: Fold per leaf, length n.
    :type fold: np.ndarray
    :param variety: Variety per leaf, length n.
    :type variety: np.ndarray
    :param w: Output of :func:`macro_weights`, shape (B, n).
    :type w: torch.Tensor
    :param device: GPU or CPU.
    :type device: torch.device
    :return: bal_acc (ensemble), boot (B,), seeds (list), by_fold (5 values) and by_variety
        ({variety: value} for varieties every task keeps; empty for one variety).
    :rtype: dict
    """
    def mean_over_tasks(fn) -> torch.Tensor:
        """Average fn(ensemble P, labels, kept leaves, seeds P) over the tasks."""
        vals = []
        for p, y in zip(p_seeds, labels):
            keep = torch.tensor(y >= 0, device=device)
            pt = torch.tensor(p, dtype=torch.float64, device=device)[:, keep]
            vals.append(fn(pt.mean(0), torch.tensor(y, dtype=torch.float64, device=device)[keep], keep, pt))
        return torch.stack(vals).mean(0)

    def part(mask: np.ndarray) -> float:
        """Macro value over the leaves in mask."""
        m = torch.tensor(mask, device=device)
        return mean_over_tasks(lambda e, y, keep, _p: balanced_accuracy(e[m[keep]], y[m[keep]])).item()

    varieties = sorted(set(variety))
    shared = [v for v in varieties if all(((variety == v) & (y >= 0)).any() for y in labels)]
    return {"bal_acc": mean_over_tasks(lambda e, y, keep, _p: balanced_accuracy(e, y)).item(),
            "boot": mean_over_tasks(lambda e, y, keep, _p: balanced_accuracy(e, y, w[:, keep])),
            "seeds": mean_over_tasks(lambda e, y, keep, p: torch.stack([balanced_accuracy(s, y) for s in p])).tolist(),
            "by_fold": [part(fold == k) for k in range(5)],
            "by_variety": {v: part(variety == v) for v in shared} if len(varieties) > 1 else {}}


def quantiles(boot: torch.Tensor) -> list[float]:
    """The 95% interval of a bootstrap distribution."""
    return torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64, device=boot.device)).tolist()
