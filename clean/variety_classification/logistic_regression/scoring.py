"""Balanced accuracy and its bootstrap, shared by logistic regression and the CNN report."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from pca_common import VARIETIES

N_BOOT, SEED = 2000, 0


def bootstrap_balanced_accuracy(y: np.ndarray, pred: np.ndarray, device: torch.device) -> list[float]:
    """Compute a 95% interval by resampling leaves with replacement.

    B: the number of resamples. n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param pred: Predicted variety codes, length n.
    :type pred: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: Lower and upper bound.
    :rtype: list[float]
    """
    g = torch.Generator(device=device).manual_seed(SEED)
    yt, pt = torch.tensor(y, device=device), torch.tensor(pred, device=device)
    idx = torch.randint(len(y), (N_BOOT, len(y)), generator=g, device=device)  # (B, n)
    y_boot, hit = yt[idx], (pt == yt)[idx]
    # Balanced accuracy = mean recall over the 4 varieties.
    recalls = torch.stack([((y_boot == k) & hit).sum(1) / (y_boot == k).sum(1)
                           for k in range(len(VARIETIES))])
    return torch.quantile(recalls.mean(0).double(), torch.tensor([0.025, 0.975], device=device,
                                                                dtype=torch.float64)).tolist()


def evaluate(y: np.ndarray, P: np.ndarray, device: torch.device) -> dict:
    """Score out-of-fold probabilities.

    Why scikit-learn: standard metric code, run once per model on 998 leaves.

    n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param P: Out-of-fold probabilities, shape (n, 4).
    :type P: np.ndarray
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: pred (n,), balanced accuracy and its CI, per-variety recall and ROC AUC, confusion (4, 4).
    :rtype: dict
    """
    pred = P.argmax(1)
    labels = range(len(VARIETIES))
    return {
        "pred": pred,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "balanced_accuracy_ci95": bootstrap_balanced_accuracy(y, pred, device),
        "recall": recall_score(y, pred, labels=labels, average=None),
        # One-vs-rest AUC: how well P(this variety) ranks its leaves above the other three.
        "auc_ovr": [roc_auc_score((y == k).astype(int), P[:, k]) for k in labels],
        "confusion": confusion_matrix(y, pred, labels=labels),
    }
