"""Leaf-level spectra shared by the analysis folders (spectra_exploration, disease_differences,
variety_logistic_regression).

    leaf_mean_spectra(meta) -> (mean, glint)   one mean spectrum per leaf + its share of glint pixels
    snv(X)                                     per-leaf brightness normalisation

The means are computed once from cache/pixels.npy (every leaf pixel, 204 bands) in float64 and
cached in cache/leaf_mean_spectra.npz, keyed by the leaf paths so a different leaf order is
never silently reused. Nothing here prints; the first call (cache missing) reads all of pixels.npy
(~6 GB) once.

SNV (standard normal variate) = subtract each spectrum's own mean and divide by its own SD over
the 204 bands; it removes brightness differences and keeps the spectral shape.
Library choice: NumPy, not torch - the cached means and the analysis results built on them were
computed this way, and swapping would change their last bits (see the "why not torch" comments).
"""
import numpy as np
import pandas as pd

from common import CACHE_DIR, load_pixels

LEAF_MEANS_PATH = CACHE_DIR / "leaf_mean_spectra.npz"


def leaf_mean_spectra(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_leaves, 204) float64 mean reflectance spectrum of every leaf in meta's order, and per leaf
    (n_leaves,) float64 the share of its pixels with any band above 1.0 (specular glints).

    Reads the cache if it was built for exactly these leaf paths in this order, otherwise computes
    it from cache/pixels.npy and (re)writes the cache."""
    if LEAF_MEANS_PATH.exists():
        d = np.load(LEAF_MEANS_PATH, allow_pickle=True)
        if list(d["path"]) == list(meta.path):
            return d["mean"], d["glint"]
    px = load_pixels(mmap=True)
    X, glint = np.empty((len(meta), px.shape[1])), np.empty(len(meta))
    # why not torch: float64 NumPy means over the memmap are what the cache holds; a torch sum order
    # would change the last bits of every cached value. One leaf at a time keeps memory small.
    for i, (s, n) in enumerate(zip(meta.px_start, meta.n_px)):
        a = np.asarray(px[s:s + n], dtype=np.float64)
        # Reflectance is at most 1 for a diffuse leaf, so a pixel above 1.0 in any band is a
        # specular glint (mirror-like reflection); its share lets analyses check glint effects.
        X[i], glint[i] = a.mean(0), (a.max(1) > 1.0).mean()
    np.savez(LEAF_MEANS_PATH, mean=X, glint=glint, path=np.array(meta.path))
    return X, glint


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate: each spectrum centred and scaled over its own bands. Removes overall
    brightness (illumination, leaf angle, glints) and keeps the spectral shape.

    X: (n_spectra, n_bands) -> same shape. Uses the population SD (ddof=0)."""
    # why not torch: torch.std defaults to the sample SD (ddof=1), NumPy's to ddof=0; keep NumPy so
    # the reported analysis numbers stay identical
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
