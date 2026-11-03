"""Within-group ROC AUC, shared by the training log and the report so their numbers agree."""
from __future__ import annotations

import numpy as np
import torch

from train import ROWS


def auc_groups(meta) -> np.ndarray:
    """Give every leaf its AUC group: variety × row.

    AUCs only compare leaves of one group, i.e. leaves of the same variety, scored by the
    same model because they share a test row, so neither recognising a variety nor a row's
    disease rate can help. For one variety, a group is simply the test row.

    :param meta: Leaf table with variety and row, n rows.
    :type meta: pandas.DataFrame
    :return: Group id per leaf, length n.
    :rtype: np.ndarray
    """
    varieties = sorted(set(meta.variety))
    variety_code = np.array([varieties.index(v) for v in meta.variety])
    return variety_code * len(ROWS) + np.array([ROWS.index(r) for r in meta.row])


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


def auc_summary(p_seeds: np.ndarray, y: np.ndarray, group: np.ndarray, fold: np.ndarray, variety: np.ndarray,
                device: torch.device) -> dict:
    """Score one model on the leaves it predicted: its seed ensemble, each seed, each fold and each variety.

    n: the leaves given. s: the number of seeds.

    :param p_seeds: P(positive) per seed and leaf, shape (s, n).
    :type p_seeds: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param group: Group per leaf from :func:`auc_groups`, length n.
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
