"""Per-photo centring. Leaves cut from one camera image (2-5 neighbouring vines) share its light;
subtracting the photo's mean feature vector (no labels; no scaling, as Baco has 2 leaves per image) removes
it. Same idea as per-row normalisation (AdaBN-style, Li et al. 2018) at the image level; no leaf paper found."""
from __future__ import annotations

import numpy as np

from harness import Candidate  # sets up the repo imports
from helpers.data import load_capture_ids
from family_spectral import FEATURES, MODELS

_CAP = {}


def capnorm(base):
    def f(F, idx):
        if "c" not in _CAP:
            _CAP["c"] = load_capture_ids(list(F["paths"]))
        X = base(F, idx).astype(np.float64)
        cap = _CAP["c"][idx]
        out = np.empty_like(X)
        for c in np.unique(cap):
            m = cap == c
            out[m] = X[m] - X[m].mean(0)
        return out
    f.__name__ = f"capnorm_{base.__name__}"
    return f


def candidates() -> list[Candidate]:
    out = []
    for feat in ("snv_mean", "rgb", "refl_vis", "snv_vis", "d1", "idx_mean"):
        for model in ("lr", "lda", "svm"):
            make, grid, how = MODELS[model]
            out.append(Candidate(f"{model}:capnorm_{feat}", capnorm(FEATURES[feat]), make, grid, how))
    return out


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates())
