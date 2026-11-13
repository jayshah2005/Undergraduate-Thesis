"""Literature family ("lit:" prefix): methods from published leaf-disease hyperspectral work, not tried before.

1. PROSPECT-D leaf traits (build_prospect.py; Féret et al. 2017, RSE 193:204-215), the plant-trait route to
   pre-visual detection of Zarco-Tejada et al. 2018 (Nature Plants 4:432-439). Feature sets: all 11 traits;
   pigments only (Cab, Car, Ant, Car/Cab, Ant/Cab); traits + SNV 510-710 nm.
2. CARS-PLS-DA: competitive adaptive reweighted sampling (Li, Liang, Xu & Cao 2009, Analytica Chimica Acta
   648:77-84) picks bands from PLS regression coefficients over 50 Monte Carlo runs; the subset with the lowest
   5-fold RMSECV is kept; PLS-DA on those bands. Band selection runs inside each training set only (many papers
   select on all samples, which leaks). Adaptation as in family_tuned.py: class-weighted logistic regression on the PLS
   scores instead of a fixed cut, so classes are balanced like every other candidate.
3. SPA-LDA: successive projections algorithm for classification (Araújo et al. 2001, Chemometrics and Intelligent
   Laboratory Systems 57:65-73; Pontes et al. 2005, same journal 78:11-18): chains of minimally collinear bands
   from every starting band, chain and length chosen by stratified 5-fold balanced error of LDA inside the
   training set. LDA with Ledoit-Wolf shrinkage and equal priors.
4. Joint 4-class model (healthy / red blotch only / leafroll only / both), the multiclass set-up of Sawyer et al.
   2023 (Frontiers in Plant Science 14:1117869): one multinomial model, P(disease) = sum of the classes that carry
   it. The other disease's label is used only to form the 4 training classes; within one variety only.

FIXED BEFORE RUNNING: the family's headline is the leave-test-row-out pick over "lit:" candidates only (within
variety: selection.py rule; Baco + Chardonnay + Vidal pooled: pooled.py rule), compared with within 0.535 / 0.554
and pooled variety-only 0.679 / 0.615. Within-row AUC is reported beside it as a diagnostic.

usage: family_literature.py <prospect|cars|spa|joint4>
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from harness import CACHE, Candidate
from family_spectral import FEATURES, MODELS

_P = {}


# ---- 1. PROSPECT-D traits ------------------------------------------------------------------------
def _traits(F):
    if not _P:
        d = np.load(CACHE / "prospect.npz")
        assert list(d["paths"]) == list(F["paths"])
        _P["t"] = d["traits"]
    return _P["t"]


def f_prospect(F, idx):
    return _traits(F)[idx]


def f_prospect_pig(F, idx):
    return _traits(F)[idx][:, [1, 2, 3, 6, 7]]


def f_prospect_snv_vis(F, idx):
    return np.concatenate([f_prospect(F, idx), FEATURES["snv_vis"](F, idx)], 1)


# ---- 2. CARS-PLS-DA ------------------------------------------------------------------------------
def cars(X: np.ndarray, y: np.ndarray, n_comp: int = 5, n_runs: int = 50, seed: int = 0) -> np.ndarray:
    """Indices of the bands CARS keeps (Li et al. 2009), X already standardised."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    a = (p / 2) ** (1 / (n_runs - 1))
    k = np.log(p / 2) / (n_runs - 1)
    keep = np.arange(p)
    subsets, rmsecv = [], []
    for i in range(1, n_runs + 1):
        cal = rng.choice(n, int(0.8 * n), replace=False)
        pls = PLSRegression(min(n_comp, len(keep)), scale=False).fit(X[cal][:, keep], y[cal])
        wts = np.abs(pls.coef_).ravel()
        wts = wts / wts.sum()
        n_keep = max(2, int(round(p * a * np.exp(-k * i))))
        top = np.argsort(-wts)[:n_keep]                                     # exponentially decreasing function
        drawn = rng.choice(top, n_keep, replace=True, p=wts[top] / wts[top].sum())   # adaptive reweighted sampling
        keep = keep[np.unique(drawn)]
        # 5-fold RMSECV of PLS on this subset
        err = 0.0
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            m = PLSRegression(min(n_comp, len(keep)), scale=False).fit(X[tr][:, keep], y[tr])
            err += ((m.predict(X[te][:, keep]).ravel() - y[te]) ** 2).sum()
        subsets.append(keep.copy())
        rmsecv.append(np.sqrt(err / n))
    return subsets[int(np.argmin(rmsecv))]


class CarsPlsDa(BaseEstimator):
    """Standardise → CARS band selection → PLS scores (n components) → class-weighted logistic regression."""

    def __init__(self, n=3):
        self.n = n

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.bands_ = cars(Z, y.astype(float))
        self.pls_ = PLSRegression(min(self.n, len(self.bands_)), scale=False).fit(Z[:, self.bands_], y)
        self.lr_ = LogisticRegression(max_iter=5000).fit(self.pls_.transform(Z[:, self.bands_]), y,
                                                          sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.pls_.transform(self.sc_.transform(X)[:, self.bands_]))


# ---- 3. SPA-LDA ----------------------------------------------------------------------------------
def spa_chain(Xc: np.ndarray, start: int, m: int) -> list[int]:
    """Successive projections from one starting band (Araújo et al. 2001): m minimally collinear bands."""
    P = Xc.copy()
    chain = [start]
    for _ in range(m - 1):
        x = P[:, chain[-1]]
        P = P - np.outer(x, x @ P) / (x @ x)            # project every column off the last chosen one
        norms = np.linalg.norm(P, axis=0)
        norms[chain] = -1
        chain.append(int(np.argmax(norms)))
    return chain


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def _bal_err(p, y):
    return 1 - 0.5 * ((p[y == 1] >= 0.5).mean() + (p[y == 0] < 0.5).mean())


class SpaLda(BaseEstimator):
    """Standardise → SPA band chain (start and length by inner stratified 5-fold balanced error) → LDA."""

    def __init__(self, max_bands=12):
        self.max_bands = max_bands

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        Zc = Z - Z.mean(0)
        m = min(self.max_bands, Z.shape[1], len(y) - 2)
        chains = [spa_chain(Zc, s, m) for s in range(Z.shape[1])]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y))
        best, best_err = None, np.inf
        for ch in chains:
            for size in range(1, m + 1):
                cols = ch[:size]
                p = np.empty(len(y))
                for tr, te in folds:
                    p[te] = _lda().fit(Z[tr][:, cols], y[tr]).predict_proba(Z[te][:, cols])[:, 1]
                e = _bal_err(p, y)
                if e < best_err - 1e-12:
                    best, best_err = cols, e
        self.bands_ = best
        self.lda_ = _lda().fit(Z[:, best], y)
        return self

    def predict_proba(self, X):
        return self.lda_.predict_proba(self.sc_.transform(X)[:, self.bands_])


# ---- 4. Joint 4-class ----------------------------------------------------------------------------
_L = {}


def with_labels(base):
    """Append each leaf's two disease labels (leafroll, red blotch) as the last 2 columns, for training classes."""
    def f(F, idx):
        if not _L:
            from logistic_regression import load_meta, task_labels   # on sys.path via harness
            meta = load_meta("all")
            assert list(meta.path) == list(F["paths"])
            _L["L"] = np.stack([task_labels(meta, "leafroll"), task_labels(meta, "red_blotch")], 1).astype(float)
        return np.concatenate([base(F, idx).astype(np.float64), _L["L"][idx]], 1)
    f.__name__ = f"{base.__name__}+labels"
    return f


class Joint4(BaseEstimator):
    """Multinomial model over (leafroll, red blotch) combinations; P(task) = sum over classes with the task.

    The last 2 columns of X are the labels: used in fit() to build the 4 classes, never in predict_proba().
    Classes are weighted equally (balanced over the combinations present), so with all 4 present the implied
    prior of each disease is 1/2 and P >= 0.5 is the balanced cut.
    """

    def __init__(self, model="lr", C=1.0):
        self.model, self.C = model, C

    def fit(self, X, y, sample_weight=None):
        F, L = X[:, :-2], X[:, -2:].astype(int)
        match = [j for j in (0, 1) if np.array_equal(L[:, j], y.astype(int))]
        assert len(match) == 1, "y must equal exactly one label column"
        self.task_ = match[0]
        cls = 2 * L[:, 0] + L[:, 1]                     # 0 healthy, 1 red blotch only, 2 leafroll only, 3 both
        present, counts = np.unique(cls, return_counts=True)
        w = (len(cls) / (len(present) * counts))[np.searchsorted(present, cls)]
        self.sc_ = StandardScaler().fit(F)
        Z = self.sc_.transform(F)
        if self.model == "lr":
            self.m_ = LogisticRegression(C=self.C, max_iter=5000).fit(Z, cls, sample_weight=w)
        else:   # LDA: equal priors over the combinations present
            self.m_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                 priors=np.full(len(present), 1 / len(present))).fit(Z, cls)
        carries = np.array([(c >> (1 - self.task_)) & 1 for c in self.m_.classes_])   # leafroll bit 2, red blotch bit 1
        self.carries_ = carries.astype(bool)
        return self

    def predict_proba(self, X):
        P = self.m_.predict_proba(self.sc_.transform(X[:, :-2]))
        p = P[:, self.carries_].sum(1)
        return np.stack([1 - p, p], 1)


# ---- candidates ----------------------------------------------------------------------------------
def candidates(part: str) -> tuple[list[Candidate], tuple]:
    if part == "prospect":
        feats = {"prospect": f_prospect, "prospect_pig": f_prospect_pig, "prospect_snv_vis": f_prospect_snv_vis}
        out = [Candidate(f"lit:{m}:{n}", fn, *MODELS[m]) for n, fn in feats.items() for m in ("lr", "lda", "svm", "rf")]
        return out, ("within", "pooled")
    if part == "cars":
        return [Candidate(f"lit:cars_plsda:{n}", FEATURES[n], CarsPlsDa, [{"n": 2}, {"n": 5}], "proba")
                for n in ("snv_mean", "refl_vis", "snv_vis")], ("within", "pooled")
    if part == "spa":
        return [Candidate(f"lit:spa_lda:{n}", FEATURES[n], SpaLda, [{}], "proba")
                for n in ("snv_mean", "refl_vis")], ("within", "pooled")
    if part == "joint4":
        feats = {"snv_vis": FEATURES["snv_vis"], "refl_vis": FEATURES["refl_vis"], "idx_mean": FEATURES["idx_mean"],
                 "prospect": f_prospect}
        out = [Candidate(f"lit:joint4_lr:{n}", with_labels(fn), Joint4, [{"C": c} for c in (1e-2, 0.1, 1.0)], "proba")
               for n, fn in feats.items()]
        out += [Candidate(f"lit:joint4_lda:{n}", with_labels(fn), Joint4, [{"model": "lda"}], "proba")
                for n, fn in feats.items()]
        return out, ("within",)
    raise ValueError(part)


if __name__ == "__main__":
    from harness import run_all
    cands, modes = candidates(sys.argv[1])
    run_all(cands, modes=modes)
