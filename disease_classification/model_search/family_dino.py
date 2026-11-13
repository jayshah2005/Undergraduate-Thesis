"""Frozen DINOv2 embeddings (Oquab et al. 2024) + linear / kernel probes."""
from __future__ import annotations

import numpy as np

from harness import CACHE, Candidate
from family_spectral import MODELS

_CACHE = {}


def dino(tag: str, render: str, part: str):
    """Feature function for one embedding file: part "cls", "mean" or "both" (concatenated)."""
    def f(F, idx):
        key = (tag, render)
        if key not in _CACHE:
            d = np.load(CACHE / f"dino_{tag}_{render}.npz")
            assert list(d["paths"]) == list(F["paths"]), "embedding order differs"
            _CACHE[key] = d
        d = _CACHE[key]
        X = {"cls": d["cls"], "mean": d["mean"], "both": np.concatenate([d["cls"], d["mean"]], 1)}[part]
        return X[idx]
    f.__name__ = f"dino_{tag}_{render}_{part}"
    return f


def candidates(tag: str = "small") -> list[Candidate]:
    out = []
    for render in ("rgb", "cir"):
        for part in ("cls", "mean", "both"):
            for model in ("lr", "pca_lr", "svm", "lda"):
                make, grid, how = MODELS[model]
                out.append(Candidate(f"{model}:dino_{tag}_{render}_{part}", dino(tag, render, part), make, grid, how))
    return out


if __name__ == "__main__":
    import sys
    from harness import run_all
    run_all(candidates(sys.argv[1] if len(sys.argv) > 1 else "small"))
