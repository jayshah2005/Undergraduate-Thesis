"""Load one mean spectrum per vine from vine_hsi.h5."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
H5_PATH = REPO_ROOT / "vine_hsi.h5"
CACHE_DIR = CLEAN_ROOT / "cache"
MEANS_PATH = CACHE_DIR / "leaf_means.npz"
META_PATH = CACHE_DIR / "meta.parquet"


def _leaf_pixels(cube: np.ndarray) -> np.ndarray:
    """Keep pixels that are leaf. Background is exactly 0 in every band.

    :param cube: Reflectance cube as stored in HDF5, shape (lines, bands, samples).
    :type cube: np.ndarray
    :return: Leaf pixels as (n_leaf_pixels, 204) float64 reflectance.
    :rtype: np.ndarray
    """
    # Why NumPy: h5py gives ndarrays. Transpose so a spatial pixel is one row of 204.
    cube_t = np.ascontiguousarray(cube.transpose(0, 2, 1))
    cube_t = np.nan_to_num(cube_t, nan=0.0)
    mask = (cube_t != 0).any(axis=2)
    return cube_t[mask].astype(np.float64) # Apply 2D mask to 3D cube and flatten into an array of all valid pixels (n_leaf_pixels, 204)

def _build() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Walk every vine in the HDF5. Slow (~1 min) only when the cache is missing.

    :return: meta, wavelengths (204,), mean spectra (n, 204), glint share (n,).
    :rtype: tuple
    """
    if not H5_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {H5_PATH}")

    rows, means, glints = [], [], []
    with h5py.File(H5_PATH, "r") as h5:
        wavelengths = np.asarray(h5["wavelengths"][:], dtype=np.float64)

        for variety in h5.keys():
            if variety == "wavelengths": 
                continue

            # Iterate over rows for each variety [Baco, Chardonnay, Vidal, Cab Franc]
            for row_name in h5[variety].keys():
                # Iterate over vines for each row
                for vine_name in h5[variety][row_name].keys():

                    # Get the data for the current vine
                    g = h5[variety][row_name][vine_name]

                    # Get the leaf pixels for the current vine
                    px = _leaf_pixels(np.asarray(g["cube"]))
                    
                    means.append(px.mean(0))    # axis 0 is the height so we are taking mean of all average reflectance values for each band and then adding them to the means list

                    # Find highest reflectance value for each leaf that are greater than 1.0 and then take the mean of all those values
                    # Glint is the proportion of leaf pixels that have a reflectance value greater than 1.0
                    glints.append(float((px.max(1) > 1.0).mean()))
                    
                    rows.append({
                        "path": f"{variety}/{row_name}/{vine_name}",
                        "variety": str(g.attrs.get("variety", variety)),
                        "row": str(g.attrs.get("row", row_name)),
                        "vine": int(g.attrs.get("vine", str(vine_name).lstrip("V"))),
                        "red_blotch": int(g.attrs.get("red_blotch", -1)),
                        "leafroll3": int(g.attrs.get("leafroll3", -1)),
                    })

    meta = pd.DataFrame(rows)
    X = np.stack(means) # Stack the means into a 2D array of shape (n_leaf_pixels, 204)

    glint = np.asarray(glints, dtype=np.float64) # Convert it to a numpy array

    # HDF5 key order is not variety/row/vine; permute so table and spectra stay aligned.
    meta["_i"] = np.arange(len(meta))
    meta = meta.sort_values(["variety", "row", "vine"]).reset_index(drop=True)  # Basically, make sure this sort does not change order
    perm = meta["_i"].to_numpy()
    meta = meta.drop(columns=["_i"])

    # Return the metadata, wavelengths, mean spectra, and glint
    return meta, wavelengths, X[perm], glint[perm]


def _save(meta: pd.DataFrame, wavelengths: np.ndarray, X: np.ndarray, glint: np.ndarray) -> None:
    """Write the leaf-mean table and spectra to cache/.

    :param meta: One row of labels per vine.
    :type meta: pd.DataFrame
    :param wavelengths: Band centres in nm, shape (204,).
    :type wavelengths: np.ndarray
    :param X: Mean reflectance, shape (n, 204).
    :type X: np.ndarray
    :param glint: Share of leaf pixels with any band > 1.0, shape (n,).
    :type glint: np.ndarray
    """
    CACHE_DIR.mkdir(exist_ok=True)
    meta.to_parquet(META_PATH)
    np.savez(MEANS_PATH, mean=X, glint=glint, wavelengths=wavelengths, path=np.array(meta.path))


def _load_cache() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the cached table, or None if it is missing or out of date.

    :return: meta, wavelengths, mean spectra, glint, or None.
    :rtype: tuple | None
    """
    if not (MEANS_PATH.exists() and META_PATH.exists()):
        return None
    meta = pd.read_parquet(META_PATH)
    d = np.load(MEANS_PATH, allow_pickle=True)
    if list(d["path"]) != list(meta.path):
        return None
    return meta, d["wavelengths"], d["mean"], d["glint"]


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return one row per vine, from cache when it is valid.

    :return: meta (998 rows), wavelengths (204,), mean (998, 204), glint (998,).
    :rtype: tuple
    """
    cached = _load_cache()
    if cached is not None:
        return cached
    meta, wavelengths, X, glint = _build()
    _save(meta, wavelengths, X, glint)
    return meta, wavelengths, X, glint
