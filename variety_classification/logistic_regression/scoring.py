"""Variety metrics and their bootstrap, shared by logistic regression and the CNN report."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from helpers.pca_common import VARIETIES
from helpers.scoring import BOOT_SEED as SEED, N_BOOT, bootstrap_weights, macro_auc  # noqa: F401


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


def row_weights(y: np.ndarray, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """The shared resampling (helpers.scoring.bootstrap_weights): leaves drawn within each row × variety cell.

    Every model scored on the same leaves gets the same resamples, so differences are paired.

    n: the number of leaves. B: the number of resamples.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf ("R1".."R5"), length n.
    :type rows: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: Resampling counts, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    group = torch.tensor([int(str(r).lstrip("R")) for r in rows], device=device)
    return bootstrap_weights(torch.tensor(y, device=device), group)


def bootstrap_metrics(y: np.ndarray, pred: np.ndarray, P: np.ndarray, rows: np.ndarray, device: torch.device) -> dict:
    """95% intervals of the overall metrics over the shared row × variety resamples.

    B: the number of resamples. n: the number of leaves. K: the number of varieties.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param pred: Predicted variety codes, length n.
    :type pred: np.ndarray
    :param P: Class probabilities (n, K), for the AUC.
    :type P: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: {metric: [lower, upper]} for accuracy, balanced_accuracy, macro_f1 and auc_macro.
    :rtype: dict
    """
    k = len(VARIETIES)
    w = row_weights(y, rows, device)                                            # (B, n)
    yt, pt = torch.tensor(y, device=device), torch.tensor(pred, device=device)
    onehot = torch.nn.functional.one_hot(yt * k + pt, k * k).double()           # (n, K*K) confusion cell of each leaf
    m = confusion_metrics((w @ onehot).view(-1, k, k))
    m["auc_macro"] = macro_auc(torch.tensor(P, dtype=torch.float64, device=device), yt, w)
    q = torch.tensor([0.025, 0.975], device=device, dtype=torch.float64)
    return {name: torch.quantile(m[name], q).tolist() for name in ("accuracy", "balanced_accuracy", "macro_f1", "auc_macro")}


def evaluate(y: np.ndarray, P: np.ndarray, rows: np.ndarray, device: torch.device) -> dict:
    """Score out-of-fold probabilities.

    Why scikit-learn: standard metric code, run once per model on 998 leaves.

    n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param P: Out-of-fold probabilities, shape (n, 4).
    :type P: np.ndarray
    :param rows: Field row per leaf (the resampling keeps each row × variety cell's size), length n.
    :type rows: np.ndarray
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: pred (n,); balanced accuracy, accuracy, macro F1 and macro AUC with CIs; per-variety recall,
        per_class (sensitivity, specificity, precision, F1) and ROC AUC; confusion (4, 4).
    :rtype: dict
    """
    pred = P.argmax(1)
    labels = range(len(VARIETIES))
    cm = confusion_matrix(y, pred, labels=labels)
    m = {k: v.tolist() for k, v in confusion_metrics(torch.tensor(cm)).items()}
    ci = bootstrap_metrics(y, pred, P, rows, device)
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
        # Their mean: how well the probabilities rank the leaves, without a cut-off.
        "auc_macro": float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in labels])),
        "auc_macro_ci95": ci["auc_macro"],
        "confusion": cm,
    }
