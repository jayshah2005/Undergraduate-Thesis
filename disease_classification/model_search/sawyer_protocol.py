"""Protocol comparison (not a result): the same models scored with Sawyer et al. 2023's split and with rows held out.

Sawyer, Laroche-Pinel, Flasco, Cooper, Corrales, Fuchs & Brillante 2023, Frontiers in Plant Science 14:1117869:
random forest on each leaf's mean 510-710 nm reflectance, "k-fold cross-validation [that] shuffled the dataset
before partitioning it into k non-overlapping folds" (leaves from any row in every fold).

Here, within each variety (and Baco + Chardonnay + Vidal pooled):
- shuffled: stratified 5-fold over leaves, 5 repeats (Sawyer's split);
- rows held out: leave-one-row-out, so each model trains on 4 rows (80% of the leaves, as a 5-fold split does).
Models: RF on 510-710 nm reflectance (Sawyer's input; 500 trees, class-balanced) and the repo baseline
(SNV leaf mean, standardised, class-balanced logistic regression, C = 1). No tuning, so no inner loop.
Scores: accuracy (what Sawyer reports) and balanced accuracy, over all out-of-fold leaves.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from family_spectral import f_refl_vis, f_snv_mean
from harness import CACHE, TASKS, VARIETIES, bal_acc, load_all
from logistic_regression import fold_of, task_labels, usable_leaves

MODELS = {
    "RF, 510-710 nm (Sawyer)": (f_refl_vis, lambda: RandomForestClassifier(500, class_weight="balanced",
                                                                           n_jobs=8, random_state=0)),
    "LR, SNV mean (repo baseline)": (f_snv_mean, lambda: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))),
}


def oof(make, X, y, splits):
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    assert not np.isnan(p).any()
    return p


def scores(p, y):
    return ((p >= 0.5) == y).mean(), bal_acc(p, y)


def compute() -> list[dict]:
    """One row per (variety or pool, task, model): accuracy and balanced accuracy under both splits."""
    data = load_all()
    table = []
    for unit in [[v] for v in VARIETIES] + [list(VARIETIES)]:
        sel = np.flatnonzero(np.isin(data["variety"], unit))
        meta = data["meta"].iloc[sel].reset_index(drop=True)
        rows = fold_of(meta)
        for task in TASKS:
            y = task_labels(meta, task)
            keep = usable_leaves(meta, y)
            idx, yk, rk = sel[keep], y[keep], rows[keep]
            for name, (feat, make) in MODELS.items():
                X = feat(data["F"], idx)
                row_splits = [(np.flatnonzero(rk != r), np.flatnonzero(rk == r)) for r in range(5)]
                sh = [scores(oof(make, X, yk, StratifiedKFold(5, shuffle=True, random_state=s).split(X, yk)), yk)
                      for s in range(5)]
                ro = scores(oof(make, X, yk, row_splits), yk)
                table.append({"leaves": " + ".join(unit), "task": task, "model": name,
                              "shuffled_acc": float(np.mean([a for a, _ in sh])),
                              "shuffled_bal_acc": float(np.mean([b for _, b in sh])),
                              "rows_acc": float(ro[0]), "rows_bal_acc": float(ro[1]),
                              "majority_acc": float(max(yk.mean(), 1 - yk.mean()))})
    return table


def main() -> None:
    """Compute the comparison and save it for report.py (a few minutes: random forests, 6 splits x 5 seeds)."""
    import json
    import warnings
    warnings.filterwarnings("ignore")
    table = compute()
    (CACHE / "sawyer_protocol.json").write_text(json.dumps(table, indent=1))
    print(f"[saved] {CACHE / 'sawyer_protocol.json'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
