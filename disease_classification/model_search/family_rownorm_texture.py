"""Per-row feature normalisation (row-wise AdaBN analogue, Li et al. 2018) and
GLCM texture (Haralick et al. 1973)."""
from __future__ import annotations

import numpy as np

from harness import CACHE, Candidate
from family_spectral import FEATURES, MODELS

_TEX = {}


def rownorm(base):
    """Standardise every feature within each variety x field row, using all of that row's leaves (no labels).

    AdaBN-style: each row (one imaging session) is normalised by its own statistics, so a row-wide shift in
    light or leaf state cannot move its leaves across the decision boundary.
    """
    def f(F, idx):
        X = base(F, idx).astype(np.float64)
        groups = np.array(["/".join(p.split("/")[:2]) for p in F["paths"][idx]])
        out = np.empty_like(X)
        for g in np.unique(groups):
            m = groups == g
            out[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
        return out
    f.__name__ = f"rownorm_{base.__name__}"
    return f


def f_texture(F, idx):
    if "X" not in _TEX:
        d = np.load(CACHE / "texture.npz")
        assert list(d["paths"]) == list(F["paths"])
        _TEX["X"] = d["X"]
    return _TEX["X"][idx]


def f_texture_idx(F, idx):
    return np.concatenate([f_texture(F, idx), FEATURES["idx"](F, idx)], 1)


def candidates() -> list[Candidate]:
    out = []
    for feat in ("snv_mean", "rgb", "refl_vis", "snv_vis", "d1", "idx", "idx_mean", "snvq"):
        for model in ("lr", "lda", "svm"):
            make, grid, how = MODELS[model]
            out.append(Candidate(f"{model}:rownorm_{feat}", rownorm(FEATURES[feat]), make, grid, how))
    for name, fn in (("texture", f_texture), ("texture_idx", f_texture_idx)):
        for model in ("lr", "pca_lr", "lda", "svm"):
            make, grid, how = MODELS[model]
            out.append(Candidate(f"{model}:{name}", fn, make, grid, how))
    return out


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates())
