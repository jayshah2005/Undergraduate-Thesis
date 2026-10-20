"""STEP 0 - read the 4 GB HDF5 file ONCE and write fast-to-load caches.

Run:  cd "claude runs/shared" && ../../.venv/bin/python -u 00_build_cache.py

Reading the gzip-compressed h5 file takes minutes, so we do it one time and
store everything the pipelines need as plain .npy files in shared/cache/:

  meta.parquet     one row per leaf: labels, variety, row, fold, pixel offsets
  pixels.npy       every LEAF pixel of every leaf, (total_pixels, 204) float32
  images.npy       whole leaves as images (N, 204, 128, 128) float16 (for CNNs)
  masks.npy        (N, 128, 128) bool, True where there is leaf
  wavelengths.npy  (204,) band centres in nm

Progress and the sanity report go to the terminal and to logs/00.log (rewritten
on every run).

What counts as a leaf pixel?  The cubes are cut-outs of single leaves with the
background set to 0, so a pixel belongs to the leaf if it is not all zeros.
This script also CHECKS the assumption that zeros are always whole pixels
(never a few bad bands inside an otherwise valid pixel) and reports the result.
"""
import logging
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (CACHE_DIR, EXP_DIR, H5_PATH, IMG_SIZE, LABELS, N_BANDS,
                    NO_LEAFROLL_VARIETY)
from logs import get_logger

LOG_FILE = EXP_DIR / "logs" / "00.log"


def to_image(cube_t: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn one leaf into a fixed 128x128 image.

    cube_t: (H, W, 204) leaf cube (background 0); mask: (H, W) bool.
    - Leaves smaller than 128 px keep their real size (centred on a black canvas),
      so absolute leaf size is preserved.
    - Leaves larger than 128 px are shrunk so the longer side is 128.
    The shrinking is "mask-aware": we interpolate (image*mask) and mask
    separately and divide, so the black background does not bleed into the
    leaf edges.
    Returns (image (204, 128, 128) float16, mask (128, 128) bool).
    """
    h, w = mask.shape
    img = torch.from_numpy(cube_t).permute(2, 0, 1)[None].float()      # (1,204,H,W)
    m = torch.from_numpy(mask)[None, None].float()                      # (1,1,H,W)
    scale = min(1.0, IMG_SIZE / max(h, w))
    if scale < 1.0:
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        num = F.interpolate(img * m, size=(nh, nw), mode="bilinear",
                            antialias=True, align_corners=False)
        den = F.interpolate(m, size=(nh, nw), mode="bilinear",
                            antialias=True, align_corners=False)
        m = (den > 0.5).float()
        img = num / den.clamp(min=1e-6) * m
        h, w = nh, nw
    canvas = torch.zeros(1, N_BANDS, IMG_SIZE, IMG_SIZE)
    mcanvas = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    t, l = (IMG_SIZE - h) // 2, (IMG_SIZE - w) // 2
    canvas[:, :, t:t + h, l:l + w] = img
    mcanvas[:, :, t:t + h, l:l + w] = m
    return canvas[0].numpy().astype(np.float16), mcanvas[0, 0].numpy() > 0.5


def leaf_paths(f: h5py.File) -> list[str]:
    """Every leaf group "variety/row/vine" in the h5, sorted by variety, row and vine NUMBER
    (V2 before V10), which fixes the leaf order of meta / pixels / images."""
    paths = [f"{v}/{r}/{n}" for v in f if v != "wavelengths"
             for r in f[v] for n in f[v][r]]
    paths.sort(key=lambda p: (p.split("/")[0], p.split("/")[1],
                              int(p.split("/")[2].lstrip("V"))))
    return paths


def read_leaf(cube: np.ndarray) -> dict:
    """Leaf pixels, 128x128 image and data-quality counts of one cube.

    cube: (lines, bands, samples) as stored in the h5 (background = 0).
    Returns px (n_leaf_px, 204) float32, img / mask from to_image, n_nan (NaN values
    replaced by 0) and n_partial (pixels zero in some but not all bands).
    """
    # why not torch: elementwise NumPy masking on one cube; identical results, and h5py hands us NumPy
    cube_t = np.ascontiguousarray(cube.transpose(0, 2, 1))   # (lines, samples, bands)
    n_nan = int(np.isnan(cube_t).sum())
    cube_t = np.nan_to_num(cube_t, nan=0.0)
    nonzero = cube_t != 0
    mask = nonzero.any(axis=2)                           # pixel is leaf if any band != 0
    n_partial = int((nonzero.any(axis=2) & ~nonzero.all(axis=2)).sum())
    px = cube_t[mask]                                    # (n_leaf_px, 204)
    img, m = to_image(cube_t, mask)
    return {"px": px.astype(np.float32), "img": img, "mask": m,
            "n_nan": n_nan, "n_partial": n_partial}


def build_cache(log: logging.Logger) -> dict:
    """Read every leaf from the h5 once and write cache/ (meta, pixels, images, masks, wavelengths).

    One progress line per 50 leaves goes to `log`. Returns meta plus the counts the sanity
    report needs (n_px_total, n_partial, n_nan, t0 = start time)."""
    CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    rows, pixel_chunks, images, masks = [], [], [], []
    n_partial = n_nan = 0
    px_start = 0

    with h5py.File(H5_PATH, "r") as f:
        np.save(CACHE_DIR / "wavelengths.npy", f["wavelengths"][:])
        paths = leaf_paths(f)
        for k, path in enumerate(paths):
            g = f[path]
            cube = g["cube"][:]                                  # (lines, bands, samples)
            leaf = read_leaf(cube)
            n_nan += leaf["n_nan"]
            n_partial += leaf["n_partial"]
            pixel_chunks.append(leaf["px"])
            images.append(leaf["img"])
            masks.append(leaf["mask"])

            variety, row, vine = path.split("/")
            rows.append({
                "path": path, "variety": variety, "row": row,
                "vine": int(vine.lstrip("V")),
                "red_blotch": int(g.attrs["red_blotch"]),
                "leafroll3": int(g.attrs["leafroll3"]),
                # fold k = rows R(k+1) of ALL varieties: whole rows are held out together because
                # neighbouring vines look alike (see common.py); every fold holds all 4 varieties
                "fold": int(row.lstrip("R")) - 1,
                "height": cube.shape[0], "width": cube.shape[2],
                "n_px": len(leaf["px"]), "px_start": px_start,
            })
            px_start += len(leaf["px"])
            if (k + 1) % 50 == 0 or k + 1 == len(paths):
                log.info(f"  {k + 1}/{len(paths)} leaves  ({time.time() - t0:.0f}s)")

    meta = pd.DataFrame(rows)
    np.save(CACHE_DIR / "pixels.npy", np.concatenate(pixel_chunks))
    np.save(CACHE_DIR / "images.npy", np.stack(images))
    np.save(CACHE_DIR / "masks.npy", np.stack(masks))
    meta.to_parquet(CACHE_DIR / "meta.parquet")
    return {"meta": meta, "n_px_total": px_start, "n_partial": n_partial, "n_nan": n_nan, "t0": t0}


def log_sanity_report(log: logging.Logger, meta: pd.DataFrame, n_px_total: int,
                      n_partial: int, n_nan: int, t0: float) -> None:
    """REPORTING: the sanity checks a user should read after building the cache (same text as
    the old printed report)."""
    log.info("\n=== sanity checks ===")
    log.info(f"leaves: {len(meta)}   leaf pixels in total: {n_px_total:,}")
    log.info(f"pixels per leaf: min {meta.n_px.min()}, median {int(meta.n_px.median())}, "
             f"max {meta.n_px.max()}")
    log.info(f"pixels that were zero in SOME but not ALL bands: {n_partial} "
             f"(0 means the 'nonzero = leaf' rule is clean)")
    log.info(f"NaN values replaced by 0: {n_nan}")
    big = (meta[["height", "width"]].max(axis=1) > IMG_SIZE).sum()
    log.info(f"leaves larger than {IMG_SIZE}px (shrunk to fit): {big}")

    meta = meta.assign(any_disease=((meta.red_blotch == 1) | (meta.leafroll3 == 1)).astype(int))
    log.info("\npositives per FOLD (rows = fold, so every fold holds all 4 varieties):")
    log.info("%s", meta.groupby("fold")[LABELS + ["any_disease"]].sum())
    log.info("\nhealthy / total per variety:")
    log.info("%s", meta.groupby("variety").apply(
        lambda d: f"{(d.any_disease == 0).sum()} / {len(d)}", include_groups=False))
    # Cab Franc is left out: it has a single leafroll-positive leaf, so leafroll is not scored there
    lr_by = meta[meta.variety != NO_LEAFROLL_VARIETY].groupby(["variety", "fold"]).leafroll3.sum()
    log.info("\nleafroll positives per (variety, fold) - a 0 here means that fold "
             "has no positive for that variety:")
    log.info("%s", lr_by.unstack())
    log.info(f"\nDone in {time.time() - t0:.0f}s. Cache in {CACHE_DIR}")


def main():
    """Set up logging -> build the cache -> report the sanity checks."""
    LOG_FILE.unlink(missing_ok=True)                 # one log per cache build, not an ever-growing file
    log = get_logger("00_build_cache", log_file=LOG_FILE)
    res = build_cache(log)
    log_sanity_report(log, **res)


if __name__ == "__main__":
    main()
