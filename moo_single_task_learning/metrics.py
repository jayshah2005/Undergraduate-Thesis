"""Score variety and disease from six-head probabilities."""
from __future__ import annotations

import numpy as np
import torch

# Why sklearn: standard ROC AUC for evaluation only (not used in training).
from sklearn.metrics import roc_auc_score

# Redefined (same meaning as data.py).
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
HEAD_NAMES = VARIETIES + ["RedBlotch", "Leafroll"]
THRESHOLD = 0.5


def balanced_accuracy(prob: torch.Tensor, y: torch.Tensor) -> float:
    """Mean of positive recall and negative recall at THRESHOLD.

    Treats both classes equally, so a rare disease is not ignored.
    Returns NaN if a class is missing in y.
    """
    y = y.double()
    pred = (prob >= THRESHOLD).double()
    pos = y == 1
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return float("nan")
    sens = (pred[pos] == 1).double().mean().item()
    spec = (pred[neg] == 0).double().mean().item()
    return 0.5 * (sens + spec)


def auc(prob: torch.Tensor, y: torch.Tensor) -> float:
    """Area under the ROC curve (0.5 = chance, 1 = perfect ranking)."""
    y_np = y.detach().cpu().numpy()
    if y_np.min() == y_np.max():
        return float("nan")
    return float(roc_auc_score(y_np, prob.detach().cpu().numpy()))


def score_binary_head(prob: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """bal_acc and AUC for one binary head."""
    return {"bal_acc": balanced_accuracy(prob, y), "auc": auc(prob, y)}


def variety_multiclass(
    logits_or_prob: torch.Tensor,
    y_onehot: torch.Tensor,
) -> dict[str, float]:
    """4-way variety score from the four variety heads.

    pred = argmax over heads 0..3. Accuracy = fraction correct.
    Balanced accuracy = mean of per-variety recalls (fair if counts differ).

    :param logits_or_prob: (n, 4) or (n, 6) — uses first 4 columns.
    :param y_onehot: (n, 4) or (n, 6) — uses first 4 columns.
    """
    scores = logits_or_prob[:, :4]
    y4 = y_onehot[:, :4]
    true_id = y4.argmax(dim=1)
    pred_id = scores.argmax(dim=1)

    acc = (pred_id == true_id).double().mean().item()

    recalls = []
    for c in range(4):
        mask = true_id == c
        if not bool(mask.any()):
            continue
        recalls.append((pred_id[mask] == c).double().mean().item())
    bal = float(sum(recalls) / len(recalls)) if recalls else float("nan")

    return {"accuracy": acc, "bal_acc": bal}


def disease_scores(
    prob: torch.Tensor,
    y: torch.Tensor,
    varieties: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Red blotch and leafroll as separate binary tasks.

    Leafroll drops Cab Franc (almost no positives in this dataset).
    Never average RB and LR into one number.
    """
    rb = score_binary_head(prob[:, 4], y[:, 4])

    keep = varieties != "Cab Franc"
    lr = score_binary_head(prob[keep, 5], y[keep, 5])

    return {"RedBlotch": rb, "Leafroll": lr}


def score_all(
    prob: torch.Tensor,
    y: torch.Tensor,
    varieties: np.ndarray,
) -> dict:
    """Full report for one test set: per-head, variety block, disease block.

    :param prob: Sigmoid probs (n, 6).
    :param y: Targets (n, 6).
    :param varieties: Variety name per leaf (length n).
    """
    per_head = {}
    for m, name in enumerate(HEAD_NAMES):
        if name == "Leafroll":
            keep = varieties != "Cab Franc"
            per_head[name] = score_binary_head(prob[keep, m], y[keep, m])
        else:
            per_head[name] = score_binary_head(prob[:, m], y[:, m])

    return {
        "per_head": per_head,
        "variety": variety_multiclass(prob, y),
        "disease": disease_scores(prob, y, varieties),
    }


if __name__ == "__main__":
    # Toy check: perfect variety probs + random disease.
    n = 40
    y = torch.zeros(n, 6)
    for i in range(n):
        y[i, i % 4] = 1.0
        y[i, 4] = float(i % 2)
        y[i, 5] = float((i // 2) % 2)
    # Perfect variety ranking; disease = labels (AUC 1).
    prob = y.clone()
    varieties = np.array([VARIETIES[i % 4] for i in range(n)])

    report = score_all(prob, y, varieties)
    print("variety", report["variety"])
    print("disease", report["disease"])
    print("per_head Baco", report["per_head"]["Baco"])