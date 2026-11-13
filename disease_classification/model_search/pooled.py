"""Pooled 3-variety disease score (Baco + Chardonnay + Vidal), with the variety's infection rate allowed to help.

Every candidate's predictions are balanced-prior posteriors (class-balanced within variety, or group-balanced
pooled). Bayes prior adjustment (Saerens, Latinne & Decaestecker 2002, Neural Computation 14(1):21-41) adds the
variety's training prior back:  logit p' = logit p + logit(pi_v) - logit(pi),  where pi_v is the variety's
positive rate in the fold's training rows and pi the pooled rate there. A leaf is positive at p' >= 0.5, which
is the pooled balanced-accuracy decision when p is calibrated.

The fold that made each prediction trained on known rows: validation row r came from fold r-1 (training rows
r+1, r+2, r+3), test row r from fold r (training rows r+2, r+3, r+4).

PROTOCOL (fixed 2026-11-13 before any pooled test score was computed)
- Score: per disease, balanced accuracy over all Baco + Chardonnay + Vidal leaves after the adjustment.
- Pool: every candidate except the early-stopped ES_ CNNs, plus "prior_only" (the variety rate alone).
- Headline: for each test row k, pick the candidate with the best pooled validation score on validation
  leaves outside row k, and score row k with it (as selection.py).
- Secondary: pick on fold k's own validation row; mean adjusted logit of the top 5 / top 10.
- Caveat: test rows were already scored inside each variety in the earlier round (README.md, Caveats).
"""
from __future__ import annotations

import numpy as np

from harness import VARIETIES


def train_rows(row: int, part: str) -> list[int]:
    """Training rows of the fold that predicted `row` as its validation or test row."""
    start = 1 if part == "val" else 2
    return [(row + start + i) % 5 for i in range(3)]


def priors(c: dict, task: str, row: int, part: str) -> tuple[dict, float]:
    """(variety positive rate, pooled positive rate) in the training rows of that fold."""
    tr = train_rows(row, part)
    pv, pos, n = {}, 0, 0
    for v in VARIETIES:
        d = c[(task, v)]
        m = np.isin(d["row"], tr)
        pv[v] = d["y"][m].mean()
        pos, n = pos + d["y"][m].sum(), n + m.sum()
    return pv, pos / n


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pooled_scores(c: dict, task: str, part: str, temp: float = 1.0, signal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjusted pooled logits, labels and varieties for one candidate (signal=False: variety prior only)."""
    s, y, var = [], [], []
    for v in VARIETIES:
        d = c[(task, v)]
        out = np.empty(len(d["y"]))
        for r in range(5):
            m = d["row"] == r
            pv, pi = priors(c, task, r, part)
            out[m] = (logit(d[part][m]) / temp if signal else 0.0) + logit(pv[v]) - logit(pi)
        s.append(out), y.append(d["y"]), var.append(np.full(len(out), v))
    return np.concatenate(s), np.concatenate(y), np.concatenate(var)


def ba(s, y):
    pred = s >= 0
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def oracle_ba(c: dict, task: str, part: str) -> float:
    """Upper bound: best per-variety thresholds chosen on the same labels (diagnostic only, never a result)."""
    P = sum((c[(task, v)]["y"] == 1).sum() for v in VARIETIES)
    N = sum((c[(task, v)]["y"] == 0).sum() for v in VARIETIES)
    gain = 0.0
    for v in VARIETIES:
        d = c[(task, v)]
        p, y = d[part], d["y"]
        best = 0.0                                   # all negative
        for t in np.concatenate([np.unique(p), [-np.inf]]):
            pred = p >= t
            best = max(best, (pred & (y == 1)).sum() / P - (pred & (y == 0)).sum() / N)
        gain += best
    return 0.5 + 0.5 * gain


def table(preds: dict, task: str) -> dict:
    """{candidate: {part: (adjusted logits, labels, rows)}} over the pooled leaves, plus prior_only."""
    out = {}
    for n, c in preds.items():
        out[n] = {}
        for part in ("val", "test"):
            s, y, _ = pooled_scores(c, task, part)
            out[n][part] = s
    c0 = next(iter(preds.values()))
    out["prior_only"] = {part: pooled_scores(c0, task, part, signal=False)[0] for part in ("val", "test")}
    y = np.concatenate([c0[(task, v)]["y"] for v in VARIETIES])
    rows = np.concatenate([c0[(task, v)]["row"] for v in VARIETIES])
    return out, y, rows


def search_scores(T: dict, y: np.ndarray, rows: np.ndarray) -> dict:
    """Headline and secondary search scores on the test rows (see PROTOCOL)."""
    names = list(T)
    res = {}
    for label, sel in (("leave-test-row-out pick", lambda k: rows != k), ("own validation row pick", lambda k: rows == (k + 1) % 5)):
        s, picks = np.full(len(y), np.nan), []
        for k in range(5):
            m = sel(k)
            best = max(names, key=lambda n: ba(T[n]["val"][m], y[m]))
            picks.append(best)
            s[rows == k] = T[best]["test"][rows == k]
        res[label] = (ba(s, y), picks)
    for kt in (5, 10):
        s = np.full(len(y), np.nan)
        for k in range(5):
            m = rows != k
            top = sorted(names, key=lambda n: -ba(T[n]["val"][m], y[m]))[:kt]
            s[rows == k] = np.mean([T[n]["test"][rows == k] for n in top], 0)
        res[f"top-{kt} mean logit"] = (ba(s, y), None)
    return res
