"""GLCM texture (Haralick, Shanmugam & Dinstein 1973) of 4 per-pixel index maps, per leaf.

Maps: ARI (1/R550 - 1/R700), NDVI (800, 670), red chromaticity (R640 / (R640 + R550 + R460)), PRI (531, 570).
Each map is quantised to 32 levels between its 1st and 99th percentile over all leaf pixels of all leaves
(labels unused); background is level 0 and its co-occurrences are dropped. Distances 1 and 3 px, 4 angles
(averaged); contrast, dissimilarity, homogeneity, ASM, energy, correlation -> 4 x 2 x 6 = 48 features.
Saved to cache/disease_search/texture.npz in load_meta("all") order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.feature import graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_cubes, load_dataset  # noqa: E402
from logistic_regression import load_meta  # noqa: E402

LEVELS = 32
PROPS = ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation")


def maps(c: np.ndarray, w: np.ndarray) -> tuple[dict, np.ndarray]:
    """Index maps (lines, samples) of one cube (204, lines, samples) and its leaf mask."""
    b = lambda nm: np.clip(c[int(np.abs(w - nm).argmin())].astype(np.float64), 1e-4, None)  # noqa: E731
    mask = (c != 0).any(0)
    rgb = b(640) + b(550) + b(460)
    return {"ARI": 1 / b(550) - 1 / b(700), "NDVI": (b(800) - b(670)) / (b(800) + b(670)),
            "chrom_r": b(640) / rgb, "PRI": (b(531) - b(570)) / (b(531) + b(570))}, mask


def main() -> None:
    meta = load_meta("all")
    w = load_dataset()[1]
    all_maps, masks, paths = [], [], []
    for v in meta.variety.unique():
        m, cubes = load_cubes(v)
        paths += list(m.path)
        for c in cubes:
            mp, mk = maps(c, w)
            all_maps.append({k: a.astype(np.float32) for k, a in mp.items()})
            masks.append(mk)
        del cubes
    assert paths == list(meta.path)
    names = list(all_maps[0])
    rng = {k: np.percentile(np.concatenate([m[k][mk] for m, mk in zip(all_maps, masks)]), [1, 99]) for k in names}
    feats = []
    for m, mk in zip(all_maps, masks):
        row = []
        for k in names:
            lo, hi = rng[k]
            q = 1 + np.floor((np.clip(m[k], lo, hi) - lo) / (hi - lo) * (LEVELS - 1e-9)).astype(np.uint8)
            q[~mk] = 0
            P = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                             levels=LEVELS + 1, symmetric=True, normed=False)[1:, 1:].astype(np.float64)
            P /= P.sum(axis=(0, 1), keepdims=True)
            for prop in PROPS:
                row += list(graycoprops(P, prop).mean(1))       # mean over angles -> one per distance
        feats.append(row)
    X = np.array(feats)
    assert np.isfinite(X).all()
    np.savez(ROOT / "cache/disease_search/texture.npz", X=X, paths=np.array(paths))
    print("[saved] texture.npz", X.shape, flush=True)


if __name__ == "__main__":
    main()
