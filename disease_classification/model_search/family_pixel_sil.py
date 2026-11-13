"""Single-instance pixel classifier (Ray & Craven 2005): every sampled pixel carries its leaf's label;
a leaf's probability aggregates its pixels' probabilities."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness import CACHE, Candidate

_PIX = {}
NPIX = 400


def pix(bin_nm: float = 10.0, lo: float = 420.0, hi: float = 980.0):
    """Feature function: each leaf's 400 pixel-SNV spectra, binned to bin_nm, flattened to (n, 400 * bins)."""
    def f(F, idx):
        if "P" not in _PIX:
            d = np.load(CACHE / "pixels.npz")
            assert list(d["paths"]) == list(F["paths"])
            _PIX["P"] = d["P"]
        P = _PIX["P"][idx].astype(np.float32)
        w = F["wavelengths"]
        edges = np.arange(lo, hi + 1e-9, bin_nm)
        B = np.stack([P[..., (w >= a) & (w < a + bin_nm)].mean(-1) for a in edges[:-1]], -1)
        return B.reshape(len(idx), -1)
    f.__name__ = f"pix{int(bin_nm)}_{int(lo)}_{int(hi)}"
    return f


class SIL:
    """Pixel-level logistic regression trained on leaf labels; leaf score = aggregate of pixel scores."""

    def __init__(self, C=1.0, agg="mean"):
        self.C, self.agg = C, agg

    def _px(self, X):
        return X.reshape(len(X) * NPIX, -1)

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        Xp = self._px(X)
        self.sc = StandardScaler().fit(Xp)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.sc.transform(Xp), np.repeat(y, NPIX), sample_weight=np.repeat(w, NPIX) / NPIX)
        return self

    def predict_proba(self, X):
        z = self.lr.decision_function(self.sc.transform(self._px(X))).reshape(len(X), NPIX)
        if self.agg == "mean":
            p = (1 / (1 + np.exp(-z))).mean(1)
        elif self.agg == "logit":
            p = 1 / (1 + np.exp(-z.mean(1)))
        else:   # "q90": a leaf is as diseased as its 10% most diseased-looking pixels
            p = np.quantile(1 / (1 + np.exp(-z)), 0.9, axis=1)
        return np.stack([1 - p, p], 1)


def candidates() -> list[Candidate]:
    out = []
    grid = [{"C": c} for c in (1e-3, 1e-2, 1e-1)]
    for feat in (pix(10.0), pix(5.0, 510.0, 710.0)):
        for agg in ("mean", "logit", "q90"):
            make = (lambda a: (lambda C: SIL(C, a)))(agg)
            out.append(Candidate(f"sil_{agg}:{feat.__name__}", feat, make, [{"C": g["C"]} for g in grid], "proba"))
    return out


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates())
