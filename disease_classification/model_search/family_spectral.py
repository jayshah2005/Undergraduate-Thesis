"""Leaf-level feature sets × classic models (the first family of the search), and the repo baseline."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from harness import Candidate

_W = None


def wavelengths(F):
    return F["wavelengths"]


def snv(A: np.ndarray) -> np.ndarray:
    """SNV over the last axis (population SD), as helpers.pca_common.snv."""
    mu = A.mean(-1, keepdims=True)
    return (A - mu) / A.std(-1, keepdims=True)


def band_range(F, lo, hi):
    w = wavelengths(F)
    return np.flatnonzero((w >= lo) & (w <= hi))


# ---- feature sets: f(F, idx) -> (n, p) -------------------------------------------------------
def f_snv_mean(F, idx):
    """SNV leaf-mean spectrum, 204 bands (the repo baseline's input)."""
    return snv(F["cached_mean"][idx])


def f_rgb(F, idx):
    """SNV over the 3 bands nearest 640 / 550 / 460 nm (the repo's RGB baseline input)."""
    w = wavelengths(F)
    b = [int(np.abs(w - nm).argmin()) for nm in (640.0, 550.0, 460.0)]
    return snv(F["cached_mean"][idx][:, b])


def f_refl_vis(F, idx):
    """Leaf-mean reflectance 510-710 nm, no SNV (Sawyer et al. 2023: RF on the visible leaf spectrum)."""
    return F["cached_mean"][idx][:, band_range(F, 510, 710)]


def f_snv_vis(F, idx):
    """SNV over 510-710 nm only."""
    return snv(F["cached_mean"][idx][:, band_range(F, 510, 710)])


def f_d1(F, idx):
    """Savitzky-Golay first derivative (window 11, order 2) of the SNV leaf mean, 420-980 nm."""
    d = savgol_filter(snv(F["cached_mean"][idx]), 11, 2, deriv=1, axis=1)
    return d[:, band_range(F, 420, 980)]


def f_idx(F, idx):
    """18 vegetation indices per pixel, summarised per leaf (mean, SD, 5/25/50/75/95th pct)."""
    return F["idx_stats"][idx].reshape(len(idx), -1)


def f_idx_mean(F, idx):
    """Leaf means of the 18 per-pixel vegetation indices."""
    return F["idx_stats"][idx][:, :, 0]


def _bin(A, F, width=20.0):
    """Average bands into width-nm bins over 420-980 nm (last axis)."""
    w = wavelengths(F)
    edges = np.arange(420, 981, width)
    cols = [A[..., (w >= a) & (w < a + width)].mean(-1) for a in edges[:-1]]
    return np.stack(cols, -1)


def f_snvq(F, idx):
    """Per-band 10/50/90th percentiles and SD of the pixel-SNV spectra, binned to 20 nm (patchy symptoms)."""
    q = _bin(F["snv_q"][idx], F)                      # (n, 3, 28)
    sd = _bin(F["snv_sd"][idx], F)                    # (n, 28)
    return np.concatenate([q.reshape(len(idx), -1), sd], 1)


def f_snvq_spread(F, idx):
    """Only the within-leaf spread: 90th - 10th percentile and SD per 20-nm bin."""
    q = _bin(F["snv_q"][idx], F)
    return np.concatenate([q[:, 2] - q[:, 0], _bin(F["snv_sd"][idx], F)], 1)


def f_idx_d1(F, idx):
    return np.concatenate([f_idx(F, idx), f_d1(F, idx)], 1)


FEATURES = {"snv_mean": f_snv_mean, "rgb": f_rgb, "refl_vis": f_refl_vis, "snv_vis": f_snv_vis, "d1": f_d1,
            "idx": f_idx, "idx_mean": f_idx_mean, "snvq": f_snvq, "snvq_spread": f_snvq_spread, "idx_d1": f_idx_d1}


# ---- models: make(**params) -> estimator ------------------------------------------------------
def lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


class _CappedPCA(PCA):
    """PCA that keeps at most as many components as the data allows (RGB has only 3 features)."""

    def fit_transform(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[0], X.shape[1])
        return super().fit_transform(X, y)

    def fit(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[0], X.shape[1])
        return super().fit(X, y)


def pca_lr(n=5):
    return make_pipeline(StandardScaler(), _CappedPCA(n_components=n), LogisticRegression(max_iter=5000))


def svm(C=1.0, gamma_mult=1.0):
    # gamma = gamma_mult / n_features on standardised inputs ("scale" for unit-variance data)
    class _S(SVC):
        def fit(self, X, y, sample_weight=None):
            self.gamma = gamma_mult / X.shape[1]
            return super().fit(X, y, sample_weight=sample_weight)
    return make_pipeline(StandardScaler(), _S(C=C, kernel="rbf"))


def rf(max_features="sqrt", min_samples_leaf=1):
    return RandomForestClassifier(n_estimators=500, max_features=max_features, min_samples_leaf=min_samples_leaf,
                                  n_jobs=14, random_state=0)


def lda():
    # Ledoit-Wolf shrinkage; equal priors so P >= 0.5 is the balanced cut.
    return make_pipeline(StandardScaler(), _LDAW(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]))


class _LDAW(LinearDiscriminantAnalysis):
    """LDA has no sample weights; equal priors already balance the classes."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y)


def hgb(max_depth=2):
    return HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                          random_state=0)


C_GRID = [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]
MODELS = {
    "lr": (lr, C_GRID, "proba"),
    "pca_lr": (pca_lr, [{"n": n} for n in (2, 3, 5, 10, 20)], "proba"),
    "svm": (svm, [{"C": c, "gamma_mult": g} for c in (0.1, 1.0, 10.0) for g in (0.3, 1.0)], "decision"),
    "rf": (rf, [{"max_features": "sqrt", "min_samples_leaf": 1}], "proba"),   # defaults: no tuning (cost)
    "lda": (lda, [{}], "proba"),
    "hgb": (hgb, [{"max_depth": 2}], "proba"),
}


def candidate(feat: str, model: str) -> Candidate:
    make, grid, how = MODELS[model]
    return Candidate(f"{model}:{feat}", FEATURES[feat], make, grid, how)


def baseline() -> Candidate:
    """The repo baseline exactly: StandardScaler + LR (C = 1, balanced) on the SNV leaf mean, no tuning."""
    return Candidate("lr_c1:snv_mean", f_snv_mean, lr, [{"C": 1.0}], "proba")


if __name__ == "__main__":
    from harness import run_all
    # Cheap models first; trees only on 4 feature sets (cost).
    order = ["lr", "lda", "pca_lr", "svm", "hgb", "rf"]
    TREE_FEATS = ["refl_vis", "snv_mean", "idx", "snvq"]
    run_all([baseline()] + [candidate(f, m) for m in order for f in (FEATURES if m not in ("rf", "hgb") else TREE_FEATS)])
