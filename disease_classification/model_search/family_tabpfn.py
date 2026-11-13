"""TabPFN v2 candidates (Hollmann et al. 2025, "Accurate predictions on small data with a tabular foundation
model", Nature 637:319-326): a transformer pre-trained on synthetic tables that predicts a new table's test
rows in one forward pass, with no tuning. Within one variety only (about 150 training leaves).

Settings: tabpfn 2.0.9 defaults except n_estimators = 8 (more preprocessing ensemble members, steadier output)
and balance_probabilities = True, which divides the output by the training class rates (balanced prior, so
P >= 0.5 is the balanced cut, like the class-weighted models). TabPFN takes no sample weights; within one
variety class balance is the only weighting the other candidates use. Raw features in: TabPFN does its own
preprocessing. Every set has <= 500 features (TabPFN v2's pre-training limit).

Needs tabpfn 2.0.9 with scikit-learn < 1.7 (it does not import with 1.9), in its own environment (README.md).

FIXED BEFORE RUNNING: TabPFN's headline is tabpfn:snv_mean@within (the repo baseline's input); the others
enter the same leave-test-row-out picks as every earlier candidate.
"""
from __future__ import annotations

import numpy as np

from harness import Candidate
from family_spectral import FEATURES
from family_dino import dino
from family_rownorm_texture import f_texture_idx, rownorm


class TabPFN:
    """sklearn-style wrapper: fit ignores sample weights, predict_proba gives the balanced posterior."""

    def __init__(self):
        from tabpfn import TabPFNClassifier
        self.m = TabPFNClassifier(device="cuda", random_state=0, n_estimators=8, balance_probabilities=True)

    def fit(self, X, y, sample_weight=None):
        assert X.shape[1] <= 500
        self.m.fit(X.astype(np.float32), y.astype(int))
        assert list(self.m.classes_) == [0, 1]
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.astype(np.float32))


def candidates() -> list[Candidate]:
    feats = {k: FEATURES[k] for k in ("snv_mean", "snv_vis", "refl_vis", "rgb", "idx", "idx_mean", "d1",
                                      "snvq", "idx_d1")}
    feats.update({f"rownorm_{k}": rownorm(FEATURES[k]) for k in ("snv_vis", "rgb", "idx_mean")})
    feats["texture_idx"] = f_texture_idx
    feats["dino_small_cir_mean"] = dino("small", "cir", "mean")
    return [Candidate(f"tabpfn:{n}", fn, TabPFN, [{}], "proba") for n, fn in feats.items()]


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates(), modes=("within",))
