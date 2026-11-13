"""Leaf shape and edge-to-centre profile candidates (build_shape.py), alone and with the 510-710 nm SNV spectrum.

Models: logistic regression, shrinkage LDA, RBF SVM, random forest (defaults); within one variety, pooled
(group-balanced), and multi-task pooled (Daume III 2007 feature augmentation, family_multitask.py).
"""
from __future__ import annotations

import numpy as np

from harness import CACHE, Candidate
from family_spectral import FEATURES, MODELS
from family_multitask import multitask

_M = {}


def _load(F):
    if not _M:
        d = np.load(CACHE / "morph.npz")
        assert list(d["paths"]) == list(F["paths"])
        _M.update({k: d[k] for k in ("shape", "profile", "zone")})
    return _M


def block(*keys):
    def f(F, idx):
        M = _load(F)
        return np.concatenate([M[k][idx] for k in keys], 1)
    f.__name__ = "_".join(keys)
    return f


def with_spectrum(base):
    def f(F, idx):
        return np.concatenate([base(F, idx), FEATURES["snv_vis"](F, idx)], 1)
    f.__name__ = f"{base.__name__}_snv_vis"
    return f


def candidates() -> list[Candidate]:
    feats = {"shape": block("shape"), "profile": block("profile"), "zone": block("zone"),
             "morph": block("shape", "profile", "zone")}
    feats["morph_snv_vis"] = with_spectrum(feats["morph"])
    out = []
    for name, fn in feats.items():
        for model in ("lr", "lda", "svm", "rf"):
            make, grid, how = MODELS[model]
            out.append(Candidate(f"{model}:{name}", fn, make, grid, how))
    mt = [Candidate(f"{m}:mt_{n}", multitask(feats[n]), *MODELS[m]) for n in ("shape", "morph") for m in ("lr", "lda")]
    return out, mt


if __name__ == "__main__":
    from harness import run_all
    out, mt = candidates()
    run_all(out)
    run_all(mt, modes=("pooled",))
