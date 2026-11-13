"""Multi-task disease models over the 3 varieties: shared + variety-specific weights.

Feature augmentation (Daume III 2007, "Frustratingly Easy Domain Adaptation", ACL, pp. 256-263): each leaf's
features x become [x, x * 1(Baco), x * 1(Chardonnay), x * 1(Vidal)], and one penalised model is fitted on the
pooled leaves. The penalty decides how much of the disease direction is shared and how much is specific to a
variety; plain pooled (shared only) and within-variety (specific only) models are the two extremes.
Runs in the harness's pooled mode: features centred per variety on training rows, group-balanced weights.
The variety's infection rate is added back afterwards by Bayes prior adjustment (pooled.py).
"""
from __future__ import annotations

import numpy as np

from harness import VARIETIES, Candidate
from family_spectral import FEATURES, MODELS
from family_rownorm_texture import rownorm


def multitask(base):
    def f(F, idx):
        X = base(F, idx).astype(np.float64)
        var = np.array([p.split("/")[0] for p in F["paths"][idx]])
        assert set(var) <= set(VARIETIES)
        return np.concatenate([X] + [X * (var == v)[:, None] for v in VARIETIES], 1)
    f.__name__ = f"mt_{base.__name__}"
    return f


def candidates() -> list[Candidate]:
    feats = {k: FEATURES[k] for k in ("snv_mean", "snv_vis", "refl_vis", "rgb", "idx", "idx_mean", "d1")}
    feats.update({f"rownorm_{k}": rownorm(FEATURES[k]) for k in ("snv_vis", "rgb", "idx_mean")})
    out = []
    for name, fn in feats.items():
        for model in ("lr", "lda", "svm"):
            make, grid, how = MODELS[model]
            out.append(Candidate(f"{model}:mt_{name}", multitask(fn), make, grid, how))
    return out


if __name__ == "__main__":
    from harness import run_all
    run_all(candidates(), modes=("pooled",))
