"""Wide hyperparameter tuning of every classical algorithm, plus PLS-DA, extra-trees and k-NN.

Tuning stays inside the training rows: the harness's inner leave-one-row-out (train on 2 training rows,
score the 3rd, mean within-variety AUC) picks each fold's configuration; the validation and test rows are
never used for it. Random configurations (Bergstra & Bengio 2012, JMLR 13:281-305) for the tree models,
which have several hyperparameters; full grids for the one- and two-parameter models. All grids are fixed
here, with a fixed seed, before any run.

FIXED BEFORE RUNNING: the tuned family's headline is the leave-test-row-out pick over the "tuned:" candidates
only (within variety: selection.py rule; 3 varieties pooled: pooled.py rule). Wide tuning raises the risk of
over-fitting the selection (Cawley & Talbot 2010, JMLR 11:2079-2107), which that pick accounts for.

Algorithms (a "tuned:" prefix marks this family):
- lr: L2, L1 and elastic-net logistic regression, C from 1e-4 to 1e2.
- pls: PLS-DA (Barker & Rayens 2003, J. Chemometrics 17(3):166-173; used for grapevine leafroll by Wang,
  Ostendorf & Pagay 2023, Sensors 23(5):2851). Adaptation: PLS components, then class-weighted logistic
  regression on them in place of a fixed 0.5 cut on the PLS regression output, so classes are balanced
  the same way as every other candidate. 1-20 components x C.
- svm: RBF (C 1e-2..1e3 x gamma) and linear SVM (C 1e-4..10).
- lda: Ledoit-Wolf and fixed shrinkage 0.01..0.95.
- rf / et: random forest and extra-trees, 16 random configurations each.
- hgb: histogram gradient boosting, 16 random configurations.
- knn: k-NN on 10 PCA components, k 5..41, uniform or distance weights; class-balanced by dividing the
  neighbour vote by the training class rates.

usage: family_tuned.py <algorithm>
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from harness import Candidate
from family_spectral import FEATURES
from family_rownorm_texture import f_texture_idx, rownorm

RNG = np.random.default_rng(20261113)


# ---- estimators -------------------------------------------------------------------------------------
def lr(penalty="l2", C=1.0, l1_ratio=None):
    solver = {"l2": "lbfgs", "l1": "liblinear", "elasticnet": "saga"}[penalty]
    kw = {"l1_ratio": l1_ratio} if penalty == "elasticnet" else {}
    return make_pipeline(StandardScaler(), LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=5000,
                                                              tol=1e-3, **kw))


class PLSScores(BaseEstimator, TransformerMixin):
    """PLS regression of the 0/1 label, used as a transformer: returns the X scores."""

    def __init__(self, n=2):
        self.n = n

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=min(self.n, X.shape[1], X.shape[0] - 1), scale=False).fit(X, y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


def pls(n=2, C=1.0):
    return make_pipeline(StandardScaler(), PLSScores(n), LogisticRegression(C=C, max_iter=5000))


def svm(kernel="rbf", C=1.0, gamma_mult=1.0):
    if kernel == "linear":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=20000))

    class _S(SVC):
        def fit(self, X, y, sample_weight=None):
            self.gamma = gamma_mult / X.shape[1]          # gamma_mult / n_features on standardised inputs
            return super().fit(X, y, sample_weight=sample_weight)
    return make_pipeline(StandardScaler(), _S(C=C, kernel="rbf"))


class _LDA(LinearDiscriminantAnalysis):
    """LDA takes no sample weights; equal priors balance the classes."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y)


def lda(shrinkage="auto"):
    return make_pipeline(StandardScaler(), _LDA(solver="lsqr", shrinkage=shrinkage, priors=[0.5, 0.5]))


def rf(**kw):
    return RandomForestClassifier(n_estimators=300, n_jobs=3, random_state=0, **kw)


def et(**kw):
    return ExtraTreesClassifier(n_estimators=300, n_jobs=3, random_state=0, **kw)


def hgb(**kw):
    return HistGradientBoostingClassifier(random_state=0, early_stopping=False, **kw)


class BalancedKNN(BaseEstimator):
    """k-NN whose neighbour vote is divided by the training class rates (balanced posterior)."""

    def __init__(self, k=15, weights="uniform"):
        self.k, self.weights = k, weights

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.pca_ = PCA(n_components=min(10, Z.shape[1], Z.shape[0])).fit(Z)
        self.knn_ = KNeighborsClassifier(n_neighbors=min(self.k, len(y) - 1), weights=self.weights)
        self.knn_.fit(self.pca_.transform(Z), y)
        self.prior_ = np.bincount(y.astype(int), minlength=2) / len(y)
        return self

    def predict_proba(self, X):
        p = self.knn_.predict_proba(self.pca_.transform(self.sc_.transform(X))) / self.prior_
        return p / p.sum(1, keepdims=True)


def knn(k=15, weights="uniform"):
    return BalancedKNN(k, weights)


# ---- grids -------------------------------------------------------------------------------------------
def random_configs(space: dict, n: int) -> list[dict]:
    out = []
    while len(out) < n:
        c = {k: v[RNG.integers(len(v))] for k, v in space.items()}
        c = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in c.items()}
        if c not in out:
            out.append(c)
    return out


C_WIDE = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
TREE_SPACE = {"max_features": ["sqrt", 0.1, 0.3, 0.6], "min_samples_leaf": [1, 3, 5, 10, 20],
              "max_depth": [None, 3, 6]}
HGB_SPACE = {"learning_rate": [0.01, 0.03, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300],
             "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0, 10.0], "max_features": [0.3, 1.0]}

ALGOS = {
    "lr": (lr, [{"penalty": "l2", "C": c} for c in C_WIDE]
           + [{"penalty": "l1", "C": c} for c in (1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)]
           + [{"penalty": "elasticnet", "C": c, "l1_ratio": 0.5} for c in (1e-2, 0.1, 1.0, 10.0)], "proba"),
    "pls": (pls, [{"n": n, "C": c} for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20) for c in (0.01, 1.0)], "proba"),
    "svm": (svm, [{"kernel": "rbf", "C": c, "gamma_mult": g} for c in (1e-2, 0.1, 1.0, 10.0, 100.0, 1e3)
                  for g in (0.03, 0.1, 0.3, 1.0, 3.0)]
            + [{"kernel": "linear", "C": c} for c in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)], "decision"),
    "lda": (lda, [{"shrinkage": s} for s in ("auto", 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)], "proba"),
    "rf": (rf, random_configs(TREE_SPACE, 16), "proba"),
    "et": (et, random_configs(TREE_SPACE, 16), "proba"),
    "hgb": (hgb, random_configs(HGB_SPACE, 16), "proba"),
    "knn": (knn, [{"k": k, "weights": w} for k in (5, 9, 15, 25, 41) for w in ("uniform", "distance")], "proba"),
}

FEATS = {k: FEATURES[k] for k in ("snv_mean", "snv_vis", "refl_vis", "d1", "idx", "idx_mean")}
FEATS["rownorm_snv_vis"] = rownorm(FEATURES["snv_vis"])
FEATS["texture_idx"] = f_texture_idx
TREE_FEATS = ("snv_vis", "refl_vis", "idx", "idx_mean")      # trees: fewer sets (cost)


def candidates(algo: str) -> list[Candidate]:
    make, grid, how = ALGOS[algo]
    feats = TREE_FEATS if algo in ("rf", "et", "hgb") else FEATS
    return [Candidate(f"tuned:{algo}:{f}", FEATS[f], make, grid, how) for f in feats]


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates(sys.argv[1]))
