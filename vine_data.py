"""Shared loaders for the vine hyperspectral HDF5 dataset."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_H5_PATH = PROJECT_DIR / "vine_hsi.h5"


def open_h5(path: Path | str = DEFAULT_H5_PATH) -> h5py.File:
    """Open the dataset read-only. Caller owns the handle."""
    return h5py.File(Path(path), "r")


def load_wavelengths(h5: h5py.File) -> np.ndarray:
    """Wavelength centres in nm, shape (204,)."""
    return h5["wavelengths"][:]


def index_vines(h5: h5py.File) -> pd.DataFrame:
    """One row per leaf/vine. Does not load cubes into memory."""
    rows = []
    for variety in h5.keys():
        if variety == "wavelengths":
            continue
        for row in h5[variety].keys():
            for vine in h5[variety][row].keys():
                g = h5[variety][row][vine]
                cube = g["cube"]
                rows.append({
                    "path": f"{variety}/{row}/{vine}",
                    "variety": g.attrs.get("variety", variety),
                    "row": g.attrs.get("row", row),
                    "vine": int(g.attrs.get("vine", vine.lstrip("V"))),
                    "ccovi_code": g.attrs.get("ccovi_code", ""),
                    "capture_date": g.attrs.get("capture_date", ""),
                    "red_blotch": int(g.attrs.get("red_blotch", -1)),
                    "leafroll3": int(g.attrs.get("leafroll3", -1)),
                    "label": np.asarray(g.attrs.get("label", []), dtype=int).tolist(),
                    "cube_shape": tuple(cube.shape),
                    "dtype": str(cube.dtype),
                })
    return (
        pd.DataFrame(rows)
        .sort_values(["variety", "row", "vine"])
        .reset_index(drop=True)
    )
