"""Bootstrap resampling and threshold-free scores shared by the variety, disease and MOO reports.

Every interval in the repo comes from the same resampling: leaves are drawn with replacement within
each (group, class) cell, where the group is the field row (× variety for a pooled disease set). The
draws are seeded, so two models scored on the same leaves see the same resamples and their
difference is paired.
"""
from __future__ import annotations

import torch

N_BOOT, BOOT_SEED = 2000, 0


def bootstrap_weights(y: torch.Tensor, group: torch.Tensor, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> torch.Tensor:
    """Resample leaves with replacement within each (group, class) cell; any number of classes.

    Keeping each cell's size fixed keeps every class's recall defined in every resample and keeps
    each row's class balance. With 0/1 labels the draws are the same, in the same order, as the
    disease scoring always made.

    n: the number of leaves. B: the number of resamples.

    :param y: Class per leaf (0/1 for a disease, 0-3 for variety), shape (n,).
    :type y: torch.Tensor
    :param group: Group per leaf (field row, or variety × row), shape (n,).
    :type group: torch.Tensor
    :param n_boot: Number of resamples B.
    :type n_boot: int
    :param seed: Generator seed.
    :type seed: int
    :return: How often each leaf is drawn in each resample, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    g = torch.Generator(device=y.device).manual_seed(seed)
    w = torch.zeros(n_boot, len(y), dtype=torch.float64, device=y.device)
    classes = torch.unique(y).tolist()
    for k in torch.unique(group).tolist():
        for c in classes:
            idx = torch.nonzero((group == k) & (y == c)).squeeze(1)
            if len(idx) == 0:
                continue
            draw = idx[torch.randint(len(idx), (n_boot, len(idx)), generator=g, device=y.device)]
            w.scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
    return w


def auc(score: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Area under the ROC curve: the chance that a random positive leaf scores above a random negative one.

    Mann-Whitney form, a tie counting one half; with resampling counts w, each leaf counts w times,
    so the result equals the AUC of the resampled leaves written out with their repeats.

    n: the number of leaves. B: the number of resamples.

    :param score: Score per leaf (e.g. P(positive)), shape (n,).
    :type score: torch.Tensor
    :param y: 0/1 label per leaf, shape (n,).
    :type y: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: AUC, shape () or (B,).
    :rtype: torch.Tensor
    """
    single = w is None
    w = torch.ones(1, len(y), dtype=torch.float64, device=y.device) if single else w
    order = torch.argsort(score)
    s, pos = score[order], (y[order] == 1).double()
    _uniq, tie = torch.unique_consecutive(s, return_inverse=True)   # tied scores share a group
    n_groups = int(tie.max()) + 1
    wo = w[:, order]
    idx = tie.expand_as(wo)
    w_pos = torch.zeros(w.shape[0], n_groups, dtype=torch.float64, device=y.device).scatter_add_(1, idx, wo * pos)
    w_neg = torch.zeros(w.shape[0], n_groups, dtype=torch.float64, device=y.device).scatter_add_(1, idx, wo * (1 - pos))
    neg_below = torch.cumsum(w_neg, 1) - w_neg   # negative weight strictly below each score group
    out = (w_pos * (neg_below + 0.5 * w_neg)).sum(1) / (w_pos.sum(1) * w_neg.sum(1))
    return out[0] if single else out


def macro_auc(P: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over classes of the one-vs-rest AUC (each class's probability ranks its leaves above the rest).

    n: the number of leaves. K: the number of classes. B: the number of resamples.

    :param P: Class probabilities, shape (n, K).
    :type P: torch.Tensor
    :param y: Class per leaf 0..K-1, shape (n,).
    :type y: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: Macro AUC, shape () or (B,).
    :rtype: torch.Tensor
    """
    return torch.stack([auc(P[:, k], (y == k).double(), w) for k in range(P.shape[1])]).mean(0)


def balanced_accuracy_multiclass(pred: torch.Tensor, y: torch.Tensor, n_classes: int,
                                 w: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over classes of the recall (the share of each class's leaves called that class).

    n: the number of leaves. B: the number of resamples.

    :param pred: Predicted class per leaf, shape (n,).
    :type pred: torch.Tensor
    :param y: True class per leaf, shape (n,).
    :type y: torch.Tensor
    :param n_classes: K.
    :type n_classes: int
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: Balanced accuracy, shape () or (B,).
    :rtype: torch.Tensor
    """
    w = torch.ones(len(y), dtype=torch.float64, device=y.device) if w is None else w
    hit = (pred == y).double()
    recalls = [(w * hit * (y == k)).sum(-1) / (w * (y == k)).sum(-1) for k in range(n_classes)]
    return torch.stack(recalls).mean(0)
