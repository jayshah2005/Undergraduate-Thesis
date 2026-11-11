"""Variety metrics and their bootstrap, shared by logistic regression and the CNN report."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from helpers.pca_common import VARIETIES

N_BOOT, SEED = 2000, 0


def confusion_metrics(cm: torch.Tensor) -> dict:
    """Every metric that follows from confusion counts (rows = true variety, columns = predicted).

    Per variety, one against the other three: sensitivity (= recall), specificity, precision, F1.
    Overall: accuracy, balanced accuracy (mean sensitivity) and macro F1 (mean F1).

    B: optional leading batch dimension (one confusion matrix per resample). K: the number of varieties.

    :param cm: Counts, shape (..., K, K).
    :type cm: torch.Tensor
    :return: Per-variety tensors (..., K) and overall tensors (...).
    :rtype: dict
    """
    cm = cm.double()
    tp = cm.diagonal(dim1=-2, dim2=-1)
    fn, fp = cm.sum(-1) - tp, cm.sum(-2) - tp
    tn = cm.sum((-2, -1))[..., None] - tp - fn - fp
    sens = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"sensitivity": sens, "specificity": tn / (tn + fp), "precision": tp / (tp + fp), "f1": f1,
            "accuracy": tp.sum(-1) / cm.sum((-2, -1)), "balanced_accuracy": sens.mean(-1), "macro_f1": f1.mean(-1)}


def bootstrap_metrics(y: np.ndarray, pred: np.ndarray, device: torch.device) -> dict:
    """95% intervals of the overall metrics, by resampling leaves with replacement.

    B: the number of resamples. n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param pred: Predicted variety codes, length n.
    :type pred: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: {metric: [lower, upper]} for accuracy, balanced_accuracy and macro_f1.
    :rtype: dict
    """
    k = len(VARIETIES)
    g = torch.Generator(device=device).manual_seed(SEED)
    yt, pt = torch.tensor(y, device=device), torch.tensor(pred, device=device)
    idx = torch.randint(len(y), (N_BOOT, len(y)), generator=g, device=device)  # (B, n)
    cell = yt[idx] * k + pt[idx]                                               # confusion cell of each draw
    cm = torch.zeros(N_BOOT, k * k, device=device, dtype=torch.float64)
    cm.scatter_add_(1, cell, torch.ones_like(cell, dtype=torch.float64))
    m = confusion_metrics(cm.view(N_BOOT, k, k))
    q = torch.tensor([0.025, 0.975], device=device, dtype=torch.float64)
    return {name: torch.quantile(m[name], q).tolist() for name in ("accuracy", "balanced_accuracy", "macro_f1")}


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
    :return: pred (n,); balanced accuracy, accuracy and macro F1 with CIs; per-variety recall,
        per_class (sensitivity, specificity, precision, F1) and ROC AUC; confusion (4, 4).
    :rtype: dict
    """
    pred = P.argmax(1)
    labels = range(len(VARIETIES))
    cm = confusion_matrix(y, pred, labels=labels)
    m = {k: v.tolist() for k, v in confusion_metrics(torch.tensor(cm)).items()}
    ci = bootstrap_metrics(y, pred, device)
    return {
        "pred": pred,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "balanced_accuracy_ci95": ci["balanced_accuracy"],
        "accuracy": m["accuracy"], "accuracy_ci95": ci["accuracy"],
        "macro_f1": m["macro_f1"], "macro_f1_ci95": ci["macro_f1"],
        "recall": recall_score(y, pred, labels=labels, average=None),
        # Per variety, one against the other three.
        "per_class": {k: m[k] for k in ("sensitivity", "specificity", "precision", "f1")},
        # One-vs-rest AUC: how well P(this variety) ranks its leaves above the other three.
        "auc_ovr": [roc_auc_score((y == k).astype(int), P[:, k]) for k in labels],
        "confusion": cm,
    }
