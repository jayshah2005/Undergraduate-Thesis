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

    lines: the number of spatial lines.
    bands: the number of spectral bands.
    samples: the number of spatial samples.
    n_leaf_pixels: the number of non-background pixels.

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

    n: the number of vines.

    :return: meta (n rows), wavelengths (204,), mean spectra (n, 204), glint share (n,).
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

    n: the number of vines.

    :param meta: Labels, n rows, one per vine.
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

    n: the number of vines.

    :return: meta (n rows), wavelengths (204,), mean spectra (n, 204), glint (n,), or None.
    :rtype: tuple | None
    """
    if not (MEANS_PATH.exists() and META_PATH.exists()):
        return None
    meta = pd.read_parquet(META_PATH)
    d = np.load(MEANS_PATH, allow_pickle=True)
    if list(d["path"]) != list(meta.path):
        return None
    return meta, d["wavelengths"], d["mean"], d["glint"]


def load_example_cube(path: str = "Baco/R1/V1"):
    """Load one vine cube for pipeline figures.

    lines, samples: spatial size of this cube.
    bands: 204.

    :param path: HDF5 group path variety/row/vine.
    :type path: str
    :return: cube (lines, samples, 204), wavelengths (204,), leaf-mean (204,), path.
    :rtype: tuple
    """
    if not H5_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {H5_PATH}")
    variety, row_name, vine_name = path.split("/")
    with h5py.File(H5_PATH, "r") as h5:
        wavelengths = np.asarray(h5["wavelengths"][:], dtype=np.float64)
        g = h5[variety][row_name][vine_name]
        # Why NumPy: h5py gives ndarrays. Spatial pixel is the last axis after this transpose.
        cube = np.ascontiguousarray(np.asarray(g["cube"]).transpose(0, 2, 1))
        cube = np.nan_to_num(cube, nan=0.0).astype(np.float64)
    mask = (cube != 0).any(axis=2)
    mean = cube[mask].mean(0)
    return cube, wavelengths, mean, path


def load_capture_ids(paths) -> np.ndarray:
    """Return the camera image (capture) each leaf was cut from.

    leaf_sources.csv matched every leaf to its capture by exact pixel values. Leaves per
    image: Baco 2, Cab Franc and Vidal 5, Chardonnay 4 in row R1 (the last image has 2)
    and 5 in rows R2-R5.

    n: the number of leaves asked for.

    :param paths: Leaf paths as in load_cubes, e.g. "Chardonnay/R1/V1", length n.
    :type paths: Iterable[str]
    :return: Capture id per leaf, length n.
    :rtype: np.ndarray
    """
    table = pd.read_csv(CLEAN_ROOT / "leaf_sources.csv")
    # "Chardonnay/R1/1/1.img" in the table is "Chardonnay/R1/V1" in the HDF5.
    parts = table.leaf.str.split("/", expand=True)
    capture = dict(zip(parts[0] + "/" + parts[1] + "/V" + parts[2], table.capture_id))
    return np.array([capture[p] for p in paths])


def load_leaf_sizes() -> tuple[pd.DataFrame, np.ndarray]:
    """Return each leaf's size, in the same order as load_dataset.

    n: the number of vines.

    :return: meta (n rows) and sizes (n, 3): leaf pixel count, height, width.
    :rtype: tuple
    """
    meta, _wavelengths, _means, _glint = load_dataset()
    sizes = np.empty((len(meta), 3))
    with h5py.File(H5_PATH, "r") as h5:
        for i, path in enumerate(meta.path):
            # Stored as (lines, bands, samples); a leaf pixel is non-zero in some band.
            cube = np.nan_to_num(np.asarray(h5[path]["cube"]), nan=0.0)
            sizes[i] = (cube != 0).any(axis=1).sum(), cube.shape[0], cube.shape[2]
    return meta, sizes


def load_cubes(variety: str) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Load every whole-leaf cube of one variety, in the same row order as load_dataset.

    n: the number of vines of this variety.
    lines, samples: spatial size of each cube (different for every leaf).

    :param variety: Variety name as in the HDF5, e.g. "Chardonnay".
    :type variety: str
    :return: meta (n rows: path, row, vine, red_blotch, leafroll3) and n cubes, each
        (204, lines, samples) float32 reflectance with background exactly 0.
    :rtype: tuple
    """
    if not H5_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {H5_PATH}")
    rows, cubes = [], []
    with h5py.File(H5_PATH, "r") as h5:
        for row_name in h5[variety].keys():
            for vine_name in h5[variety][row_name].keys():
                g = h5[variety][row_name][vine_name]
                # Why NumPy: h5py gives ndarrays. Bands first, as torch convolutions expect (C, H, W).
                cube = np.asarray(g["cube"], dtype=np.float32).transpose(1, 0, 2)
                cubes.append(np.ascontiguousarray(np.nan_to_num(cube, nan=0.0)))
                rows.append({
                    "path": f"{variety}/{row_name}/{vine_name}",
                    "row": str(g.attrs.get("row", row_name)),
                    "vine": int(g.attrs.get("vine", str(vine_name).lstrip("V"))),
                    "red_blotch": int(g.attrs.get("red_blotch", -1)),
                    "leafroll3": int(g.attrs.get("leafroll3", -1)),
                })
    meta = pd.DataFrame(rows)
    # Same order as load_dataset (variety, row, vine), so leaf i matches its mean spectrum.
    order = meta.sort_values(["row", "vine"]).index.to_numpy()
    return meta.loc[order].reset_index(drop=True), [cubes[i] for i in order]


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
