"""Tuned TabPFN v2 (Hollmann et al. 2025): n_estimators, softmax temperature and averaging before or after
the softmax, chosen per fold by the harness's inner leave-one-row-out on the training rows (family_tuned.py rule).
Within one variety. Joins the "tuned:" family (same fixed headline rule as family_tuned.py).
Needs tabpfn 2.0.9 with scikit-learn < 1.7, in its own environment (README.md).
"""
from __future__ import annotations

import numpy as np

from harness import Candidate
from family_tuned import FEATS


class TabPFN:
    """sklearn-style wrapper; balanced posterior (balance_probabilities), sample weights unused."""

    def __init__(self, n_estimators=4, softmax_temperature=0.9, average_before_softmax=False):
        from tabpfn import TabPFNClassifier
        self.m = TabPFNClassifier(device="cuda", random_state=0, balance_probabilities=True, n_estimators=n_estimators,
                                  softmax_temperature=softmax_temperature, average_before_softmax=average_before_softmax)

    def fit(self, X, y, sample_weight=None):
        assert X.shape[1] <= 500
        self.m.fit(X.astype(np.float32), y.astype(int))
        assert list(self.m.classes_) == [0, 1]
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.astype(np.float32))


GRID = [{"n_estimators": n, "softmax_temperature": t, "average_before_softmax": a}
        for n in (4, 16) for t in (0.5, 0.9, 1.5) for a in (False, True)]


def candidates() -> list[Candidate]:
    return [Candidate(f"tuned:tabpfn:{f}", FEATS[f], TabPFN, GRID, "proba")
            for f in ("snv_vis", "refl_vis", "idx_mean", "texture_idx")]


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates(), modes=("within",))
