"""Shared harness of the disease model search: data, row folds, fitting one candidate, saving its predictions.

PROTOCOL (fixed before any candidate ran)
- Targets: leafroll and red blotch, inside Baco, Chardonnay and Vidal.
- Folds: the repo's (test row k, validation row k+1, train the other 3 rows), usable_leaves per task.
- Each candidate stores P(positive) per leaf twice: from the fold whose validation row holds the leaf,
  and from the fold whose test row holds it.
- Hyperparameters: inner leave-one-row-out over the 3 training rows (mean within-variety AUC), never the
  validation row. Transforms (scaler, PCA, per-variety centring) are fitted on training rows only.
- Class balance everywhere (the score thresholds at P >= 0.5). Pooled training balances class within each
  variety (group balancing, Idrissi et al. 2022), so the variety alone predicts nothing.
- Selection score: per disease, mean over the 3 varieties of validation balanced accuracy.
- Honest search score: for each test row k, pick the candidate on validation leaves outside row k only,
  then score row k with that pick (selection.py).

Why NumPy and scikit-learn rather than PyTorch: every non-CNN candidate is a scikit-learn model, fitted thousands of
times on 150-750 leaves; the CNN family (family_cnn.py) uses PyTorch on the GPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.data import load_dataset  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from logistic_regression import fold_of, fold_split, load_meta, task_labels, usable_leaves  # noqa: E402

CACHE = ROOT / "cache/disease_search"
PRED = CACHE / "preds"
VARIETIES = ("Baco", "Chardonnay", "Vidal")
TASKS = ("leafroll", "red_blotch")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_all() -> dict:
    """Leaf table of all 998 leaves plus the cached feature arrays, same order."""
    meta = load_meta("all")
    F = dict(np.load(CACHE / "features.npz"))
    assert list(F["paths"]) == list(meta.path), "feature order differs from load_meta"
    # The baseline's exact input: helpers.data's cached leaf means (same order: all 998 leaves).
    dmeta, _w, means, _g = load_dataset()
    assert list(dmeta.path) == list(meta.path)
    F["cached_mean"] = means
    return {"meta": meta, "F": F, "variety": meta.variety.to_numpy(), "fold": fold_of(meta)}


def group_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Weight n / (n_groups_classes · n_(group, class)): every (group, class) cell gets the same total weight.

    With one group this is scikit-learn's class_weight="balanced".
    """
    w = np.zeros(len(y))
    cells = {(g, c) for g, c in zip(groups, y)}
    for g, c in cells:
        m = (groups == g) & (y == c)
        w[m] = len(y) / (len(cells) * m.sum())
    return w


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def bal_acc(p: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy of P >= 0.5."""
    pred = p >= 0.5
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def auc(s: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC (rank-based; ties count half)."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------
# Fitting one candidate on one split
# --------------------------------------------------------------------------
class Candidate:
    """A feature function plus a model factory with a small hyperparameter grid.

    features(F, idx) -> (n, p) matrix. make(params) -> unfitted estimator with fit(X, y, sample_weight)
    and a score(X) giving P(positive) (or a monotone score where 0.5 is the balanced cut).
    """

    def __init__(self, name, features, make, grid=({},), score="proba"):
        self.name, self.features, self.make, self.grid, self.score = name, features, make, list(grid), score

    def fit_predict(self, Xtr, ytr, wtr, Xte, params):
        est = self.make(**params)
        fit(est, Xtr, ytr, wtr)
        return predict(est, Xte, self.score)


def fit(est, X, y, w):
    """Fit with sample weights, routed to the last step of a pipeline."""
    from sklearn.pipeline import Pipeline
    if isinstance(est, Pipeline):
        est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": w})
    else:
        est.fit(X, y, sample_weight=w)


def predict(est, X, how):
    """P(positive); for decision functions, the logistic of the margin (0 margin -> 0.5)."""
    if how == "proba":
        return est.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-est.decision_function(X)))


def centre_by_variety(X, var, tr, y):
    """Subtract each variety's class-balanced training centre and divide by its training SD.

    The centre is the mean of that variety's positive and negative training means, so a variety's
    disease rate does not move it. Uses training rows only.
    """
    Xc = X.astype(np.float64).copy()
    for v in np.unique(var):
        t = tr[var[tr] == v]
        mu = 0.5 * (X[t][y[t] == 1].mean(0) + X[t][y[t] == 0].mean(0))
        sd = X[t].std(0) + 1e-9
        m = var == v
        Xc[m] = (X[m] - mu) / sd
    return Xc


def run_split(cand, X, y, var, rows, tr, te_list, pooled):
    """Tune on inner leave-one-row-out of the training rows, refit on all of them, predict te_list.

    Pooled: features are centred per variety on whichever rows a model trains on (raw X in, never
    centred twice).
    """
    w = group_weights(y[tr], var[tr])
    best, best_s = cand.grid[0], -np.inf
    if len(cand.grid) > 1:
        tr_rows = np.unique(rows[tr])
        for params in cand.grid:
            s = np.zeros(len(tr))
            for r in tr_rows:
                inner_te = rows[tr] == r
                inner_tr = ~inner_te
                itr = tr[inner_tr]
                Xin = centre_by_variety(X, var, itr, y) if pooled else X   # centring from the inner rows only
                s[inner_te] = cand.fit_predict(Xin[itr], y[itr], group_weights(y[itr], var[itr]),
                                               Xin[tr[inner_te]], params)
            # Mean AUC inside each variety, so pooled tuning targets the within-variety score.
            score = np.mean([auc(s[var[tr] == v], y[tr][var[tr] == v]) for v in np.unique(var[tr])])
            if score > best_s:
                best, best_s = params, score
    Xo = centre_by_variety(X, var, tr, y) if pooled else X
    return [cand.fit_predict(Xo[tr], y[tr], w, Xo[te], best) for te in te_list], best


def evaluate(cand, data, mode: str) -> dict:
    """Run one candidate over both tasks and 5 folds; mode "within" (one model per variety) or "pooled".

    :return: {(task, variety): {"val": P per leaf, "test": P per leaf, "y": labels, "row": fold}} over
        that variety's usable leaves, plus the chosen hyperparameters.
    """
    out, chosen = {}, []
    for task in TASKS:
        units = [[v] for v in VARIETIES] if mode == "within" else [list(VARIETIES)]
        for vs in units:
            sel = np.isin(data["variety"], vs)
            meta = data["meta"][sel].reset_index(drop=True)
            y = task_labels(meta, task)
            keep = usable_leaves(meta, y)
            fold, var = fold_of(meta), meta.variety.to_numpy()
            X = cand.features(data["F"], np.flatnonzero(sel))
            val_p, test_p = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for k in range(5):
                tr, va, te = fold_split(fold, k, keep)
                (pv, pt), params = run_split(cand, X, y, var, fold, tr, [va, te], mode == "pooled")
                val_p[va], test_p[te] = pv, pt
                chosen.append(params)
            for v in vs:
                m = keep & (var == v)
                assert not np.isnan(val_p[m]).any() and not np.isnan(test_p[m]).any()
                out[(task, v)] = {"val": val_p[m], "test": test_p[m], "y": y[m], "row": fold[m]}
    return {"preds": out, "params": chosen}


def save(name: str, res: dict) -> None:
    """Store a candidate's predictions in cache/disease_search/preds/<name>.npz."""
    PRED.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for (task, v), d in res["preds"].items():
        for k, a in d.items():
            arrs[f"{task}|{v}|{k}"] = a
    np.savez(PRED / f"{name}.npz", **arrs, params=np.array([str(p) for p in res["params"]]))


def summary(res: dict, show_test: bool = False) -> str:
    """One line: validation balanced accuracy per task (mean, then Baco Chardonnay Vidal).

    Test scores stay hidden while the search runs, so they cannot steer it.
    """
    parts = []
    for task in TASKS:
        vals = [bal_acc(res["preds"][(task, v)]["val"], res["preds"][(task, v)]["y"]) for v in VARIETIES]
        line = f"{task[:2]} val {np.mean(vals):.3f} [" + " ".join(f"{x:.2f}" for x in vals) + "]"
        if show_test:
            tests = [bal_acc(res["preds"][(task, v)]["test"], res["preds"][(task, v)]["y"]) for v in VARIETIES]
            line += " test [" + " ".join(f"{x:.3f}" for x in tests) + "]"
        parts.append(line)
    return " | ".join(parts)


def run_all(cands: list, modes=("within", "pooled"), skip_done=True) -> None:
    """Evaluate and save every (candidate, mode) not already on disk; log one validation line each.

    Called from each family's main(): it sets up the logger, the only output besides the prediction files.
    """
    import warnings
    warnings.filterwarnings("ignore")   # convergence and deprecation chatter from scikit-learn
    log = get_logger("model_search")
    data = load_all()
    for c in cands:
        for mode in modes:
            name = f"{c.name}@{mode}"
            if skip_done and (PRED / f"{name}.npz").exists():
                continue
            t0 = time.time()
            res = evaluate(c, data, mode)
            save(name, res)
            log.info("%-40s %s  (%.0f s)", name, summary(res), time.time() - t0)
