"""Load vines, build six-head labels, row folds, and padded SNV batches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
H5_PATH = ROOT / "vine_hsi.h5"

# Redefined here on purpose (not imported from helpers).
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
ROWS = ["R1", "R2", "R3", "R4", "R5"]
HEAD_NAMES = VARIETIES + ["RedBlotch", "Leafroll"]
N_HEADS = 6
CANVAS = 211  # largest leaf side in this HDF5


@dataclass
class VineData:
    """All leaves needed for training and scoring."""

    paths: list[str]
    varieties: np.ndarray   # name per leaf
    rows: np.ndarray        # "R1"…"R5"
    fold_ids: np.ndarray    # 0..4
    y: np.ndarray           # (n, 6) float32
    cubes: list[np.ndarray] # each (204, H, W) float32, background 0
    canvas: int = CANVAS


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_one_vine(h5: h5py.File, path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one cube and its six targets.

    HDF5 stores (lines, bands, samples). We transpose to (bands, H, W) for Conv2d.
    y = [Baco, CabFranc, Chardonnay, Vidal, red_blotch, leafroll].
    """
    variety, row, vine = path.split("/")
    g = h5[variety][row][vine]

    cube = np.asarray(g["cube"], dtype=np.float32).transpose(1, 0, 2)
    cube = np.nan_to_num(cube, nan=0.0)

    y = np.zeros(N_HEADS, dtype=np.float32)
    y[VARIETIES.index(variety)] = 1.0
    y[4] = float(g.attrs.get("red_blotch", 0))
    y[5] = float(g.attrs.get("leafroll3", 0))
    return cube, y


def list_all_paths() -> list[str]:
    """Every variety/row/vine path in a stable order."""
    paths: list[str] = []
    with h5py.File(H5_PATH, "r") as h5:
        for variety in VARIETIES:
            for row in ROWS:
                vines = sorted(
                    h5[variety][row].keys(),
                    key=lambda v: int(str(v).lstrip("V")),
                )
                for vine in vines:
                    paths.append(f"{variety}/{row}/{vine}")
    return paths


def load_vines() -> VineData:
    """Load all ~998 cubes and labels into memory (unpadded)."""
    paths = list_all_paths()
    cubes: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rows: list[str] = []
    varieties: list[str] = []

    with h5py.File(H5_PATH, "r") as h5:
        for path in paths:
            cube, y = load_one_vine(h5, path)
            variety, row, _ = path.split("/")
            cubes.append(cube)
            ys.append(y)
            rows.append(row)
            varieties.append(variety)

    row_arr = np.asarray(rows)
    return VineData(
        paths=paths,
        varieties=np.asarray(varieties),
        rows=row_arr,
        fold_ids=np.array([ROWS.index(r) for r in row_arr], dtype=np.int64),
        y=np.stack(ys),
        cubes=cubes,
        canvas=CANVAS,
    )


def fold_split(fold_ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = row k, validation = row (k+1) mod 5, train = the other three.

    Whole rows are held out because neighbouring vines look alike.
    """
    idx = np.arange(len(fold_ids))
    val = (k + 1) % 5
    test = fold_ids == k
    valid = fold_ids == val
    train = (~test) & (~valid)
    return idx[train], idx[valid], idx[test]


def pad_batch(
    cubes: list[np.ndarray],
    side: int = CANVAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre each (204, H, W) leaf on a square canvas; mask = 1 on leaf pixels."""
    assert side >= max(max(c.shape[1], c.shape[2]) for c in cubes)

    b = len(cubes)
    x = np.zeros((b, 204, side, side), dtype=np.float32)
    for i, c in enumerate(cubes):
        _, h, w = c.shape
        top = (side - h) // 2
        left = (side - w) // 2
        x[i, :, top:top + h, left:left + w] = c

    mask = (x != 0).any(axis=1, keepdims=True).astype(np.float32)
    return x, mask


def snv_cube(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel SNV over bands, then zero the background.

    SNV = (spectrum - mean) / std so brightness drops out and shape remains.
    """
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return ((x - mu) / sd) * mask


def make_batch(
    data: VineData,
    indices: np.ndarray | list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad + SNV one mini-batch onto ``device``.

    Returns X (B, 204, S, S), mask (B, 1, S, S), Y (B, 6).
    """
    idx = np.asarray(indices)
    x_np, mask_np = pad_batch([data.cubes[i] for i in idx], side=data.canvas)
    x = torch.as_tensor(x_np, device=device)
    mask = torch.as_tensor(mask_np, device=device)
    y = torch.as_tensor(data.y[idx], device=device)
    return snv_cube(x, mask), mask, y


if __name__ == "__main__":
    device = get_device()
    data = load_vines()
    tr, va, te = fold_split(data.fold_ids, 0)
    x, mask, y = make_batch(data, te[:4], device)
    
    print("n", len(data.paths), "canvas", data.canvas, "device", device)
    print("fold0 train/val/test", len(tr), len(va), len(te))
    print("batch X", tuple(x.shape), "Y", tuple(y.shape))
    print("first test Y", y[0].tolist())
    print(
        "variety one-hot sums min/max",
        float(data.y[:, :4].sum(1).min()),
        float(data.y[:, :4].sum(1).max()),
    )