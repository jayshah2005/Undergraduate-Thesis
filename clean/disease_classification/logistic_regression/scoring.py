"""Within-group ROC AUC and its bootstrap, shared by the baseline, the CNN's training log and the CNN report."""
from __future__ import annotations

import numpy as np
import torch

N_BOOT, BOOT_SEED = 2000, 0


def _pair_tables(p: torch.Tensor, y: torch.Tensor, group: torch.Tensor) -> list:
    """Per group: 1 / 0.5 / 0 for each (positive, negative) pair, by who scores higher."""
    tabs = []
    for k in torch.unique(group).tolist():
        pk, yk = p[group == k], y[group == k]
        pos, neg = pk[yk == 1], pk[yk == 0]
        tabs.append((k, (pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()))
    return tabs


def within_row_auc(p: torch.Tensor, y: torch.Tensor, group: torch.Tensor,
                   w: torch.Tensor | None = None) -> torch.Tensor:
    """ROC AUC that only compares leaves of the same group (same variety and row).

    This builds on the archived study's "within-variety AUC":
    a leaf is compared only with leaves scored by the same model, so differences
    between the five models (e.g. a shifted threshold) cannot help or hurt; and only
    with leaves of its own variety and row, so neither recognising a variety nor a
    row's disease rate can help.
    Groups with no positive or no negative leaf contribute no pairs.
    With bootstrap weights w (B, n), returns one AUC per bootstrap sample.

    :param p: Probabilities, shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param group: Group id per leaf (variety × row), shape (n,).
    :type group: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: AUC, shape () or (B,).
    :rtype: torch.Tensor
    """
    num = den = 0.0
    for k, tab in _pair_tables(p, y, group):
        mk = group == k
        if w is None:
            num, den = num + tab.sum(), den + tab.numel()
        else:
            wk, yk = w[:, mk], y[mk]
            wp, wn = wk[:, yk == 1], wk[:, yk == 0]
            num = num + ((wp @ tab) * wn).sum(1)
            den = den + wp.sum(1) * wn.sum(1)
    return num / den


def bootstrap_weights(y: torch.Tensor, group: torch.Tensor, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> torch.Tensor:
    """Resample leaves with replacement within each (group, class) cell.

    Keeping each cell's size fixed keeps every bootstrap AUC defined and leaves
    the group structure intact. Seeded so every model sees the same resamples, which
    makes the model-minus-baseline differences paired.

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


def auc_summary(p_seeds: np.ndarray, y: np.ndarray, group: np.ndarray, fold: np.ndarray, variety: np.ndarray,
                device: torch.device) -> dict:
    """Score one model on the leaves it predicted: its seed ensemble, each seed, each fold and each variety.

    n: the leaves given. s: the number of seeds.

    :param p_seeds: P(positive) per seed and leaf, shape (s, n).
    :type p_seeds: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param group: AUC group per leaf (variety × row), length n.
    :type group: np.ndarray
    :param fold: Fold per leaf, length n.
    :type fold: np.ndarray
    :param variety: Variety per leaf, length n.
    :type variety: np.ndarray
    :param device: GPU or CPU.
    :type device: torch.device
    :return: auc (ensemble), seeds (list), by_fold ({fold: AUC}) and by_variety ({variety: AUC}, empty for one variety).
    :rtype: dict
    """
    p = torch.tensor(p_seeds, dtype=torch.float64, device=device)
    labels = torch.tensor(y, dtype=torch.float64, device=device)
    groups = torch.tensor(group, device=device)
    ens = p.mean(0)

    def auc(mask: np.ndarray) -> float:
        """Ensemble AUC over the leaves in mask."""
        m = torch.tensor(mask, device=device)
        return within_row_auc(ens[m], labels[m], groups[m]).item()

    varieties = sorted(set(variety))
    return {"auc": within_row_auc(ens, labels, groups).item(),
            "seeds": [within_row_auc(s, labels, groups).item() for s in p],
            "by_fold": {int(k): auc(fold == k) for k in np.unique(fold)},
            "by_variety": {v: auc(variety == v) for v in varieties} if len(varieties) > 1 else {}}
