"""Shared helpers used by every experiment script.

WHY THIS FILE EXISTS
--------------------
All approaches must be evaluated in exactly the same way, otherwise their
scores cannot be compared. So every script imports its data loading, its
train/test folds and its metrics from here.

THE EVALUATION PROTOCOL (the part that matters most)
----------------------------------------------------
* Unit of classification = one leaf ("vine"). 998 leaves, two binary labels:
  red_blotch and leafroll3 (a leaf can have both, or neither = healthy).
* 5-fold cross-validation where fold k = every leaf in rows R(k+1), across
  all four varieties. Neighbouring vines look alike, so whole rows are held
  out together; a random split would leak and give inflated scores.
* Pixels are NEVER split across folds: all pixels of a leaf live in the same
  fold as the leaf (pixels of one leaf are near-copies of each other).
* Scalers / PCA / etc. are always fitted on the training folds only.
* Cab Franc has exactly 1 leafroll-positive leaf, so it is excluded from the
  leafroll loss and from the leafroll metrics (it is still used for red blotch).
* "Within-variety" AUCs compare a leaf only with leaves of the same variety and
  fold (stratified AUC), so a model that merely recognises the variety scores 0.5.
* Each approach writes out-of-fold (OOF) probabilities for every leaf to
  results/<name>_oof.csv (save_oof here, or four_class.save_results for the
  4-class pipelines); the archived study's compare script turned those files
  into one table.

USE
---
    from common import load_meta, n_folds, fold_indices, evaluate
    meta = load_meta()
    for fold in range(n_folds()):
        train_idx, test_idx = fold_indices(meta, fold)

Reads shared/cache/ (built by 00_build_cache.py). Metric functions return
numbers / dicts and never print; save_oof and print_fold_summary are the only
reporting helpers (they print, or write to a logger passed as `log=`).

Library choice (prefer-pytorch skill): this module is metric / I/O code on
~1000 leaves, so it stays in NumPy / pandas / sklearn.metrics; the non-torch
numerical calls that are not plain metrics or I/O carry a "why not torch" comment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent            # claude runs/shared/
PROJECT_DIR = EXP_DIR.parent.parent                  # Classification/ (repo root)
H5_PATH = PROJECT_DIR / "vine_hsi.h5"
CACHE_DIR = EXP_DIR / "cache"          # written once by 00_build_cache.py
RESULTS_DIR = EXP_DIR / "results"      # default output folder of save_results (the original 11_pmmfnet.py writes here)

N_BANDS = 204
IMG_SIZE = 128                          # whole-leaf images are padded to 128x128
N_FOLDS = 5
LABELS = ["red_blotch", "leafroll3"]
NO_LEAFROLL_VARIETY = "Cab Franc"       # only 1 positive -> leafroll undefined


def _report(msg: str, log: logging.Logger | None) -> None:
    """Send one progress line to `log` if given, else print it (the historic behaviour)."""
    (log.info if log is not None else print)(msg)


# --------------------------------------------------------------------------
# Loading the cache built by 00_build_cache.py
# --------------------------------------------------------------------------
def load_meta() -> pd.DataFrame:
    """One row per leaf (998 rows, sorted by variety, row, vine): path, variety, row, vine,
    red_blotch, leafroll3, fold, height, width, n_px, px_start."""
    return pd.read_parquet(CACHE_DIR / "meta.parquet")


def load_pixels(mmap: bool = False) -> np.ndarray:
    """All leaf pixels of all leaves stacked: shape (total_pixels, 204), float32.

    Leaf i owns rows meta.px_start[i] : meta.px_start[i] + meta.n_px[i].
    mmap=True avoids reading all ~8 GB into RAM (slower random access).
    """
    return np.load(CACHE_DIR / "pixels.npy", mmap_mode="r" if mmap else None)


def load_images(mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Whole-leaf images for CNNs.

    images: (N_leaves, 204, 128, 128) float16, background = exactly 0
    masks : (N_leaves, 128, 128) bool, True where there is leaf
    """
    mode = "r" if mmap else None
    return (np.load(CACHE_DIR / "images.npy", mmap_mode=mode),
            np.load(CACHE_DIR / "masks.npy", mmap_mode=mode))


def load_wavelengths() -> np.ndarray:
    """Band centres in nm, (204,)."""
    return np.load(CACHE_DIR / "wavelengths.npy")


def leaf_pixels(pixels: np.ndarray, meta: pd.DataFrame, i: int) -> np.ndarray:
    """All pixels (n, 204) of leaf number i (row position in meta)."""
    s = int(meta.px_start.iloc[i])
    return pixels[s:s + int(meta.n_px.iloc[i])]


# --------------------------------------------------------------------------
# Labels and folds
# --------------------------------------------------------------------------
def label_matrix(meta: pd.DataFrame) -> np.ndarray:
    """(N, 2) float array: columns = [red_blotch, leafroll3]."""
    return meta[LABELS].to_numpy(dtype=np.float32, copy=True)


def label_mask(meta: pd.DataFrame) -> np.ndarray:
    """(N, 2) bool array: True where a label may be used for training/scoring.

    Leafroll is switched off for Cab Franc (1 positive in 250 leaves): with a
    single positive the label cannot be learned or scored, and keeping it would
    teach the model "Cab Franc => no leafroll", a variety shortcut.
    """
    m = np.ones((len(meta), 2), dtype=bool)
    m[(meta.variety == NO_LEAFROLL_VARIETY).to_numpy(), 1] = False
    return m


VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]


def n_folds(scheme: str = "row") -> int:
    """Number of folds: 5 rows ('row'), or 4 varieties ('lovo', leave-one-variety-out)."""
    return len(VARIETIES) if scheme == "lovo" else N_FOLDS


def fold_indices(meta: pd.DataFrame, fold: int,
                 scheme: str = "row") -> tuple[np.ndarray, np.ndarray]:
    """Row positions (train_idx, test_idx) for one fold.

    scheme="row"  : the standard 5 folds (fold k = rows R(k+1) of all varieties).
                    Whole rows are held out because neighbouring vines look
                    alike; splitting a row between train and test would leak
                    and inflate scores.
    scheme="lovo" : leave-one-variety-out, 4 folds. The test fold is a whole
                    variety the model has never seen - the hardest stress test
                    for "is it learning disease or just the variety?".
    Both are int64 index arrays into meta (positions, not labels), sorted.
    """
    if scheme == "lovo":
        is_test = (meta.variety == VARIETIES[fold]).to_numpy()
    else:
        is_test = (meta.fold == fold).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


# --------------------------------------------------------------------------
# Pixel sampling and per-band standardisation (always from TRAIN leaves only)
# --------------------------------------------------------------------------
def sample_pixels_per_leaf(pixels, meta, leaf_idx, k, seed=0):
    """Draw up to k random pixels from each leaf in leaf_idx.

    Returns (X, owner): X is (n, 204) float32, owner[j] = position in meta of
    the leaf pixel j came from (use it to look up the leaf's label).
    Sampling a fixed number per leaf keeps big leaves from dominating.
    """
    # why not torch: the archived results depend on this exact NumPy random stream
    rng = np.random.default_rng(seed)
    xs, owners = [], []
    for i in leaf_idx:
        n = int(meta.n_px.iloc[i])
        s = int(meta.px_start.iloc[i])
        pick = rng.choice(n, size=min(k, n), replace=False)
        pick.sort()                       # sorted reads are faster on a memmap
        xs.append(np.asarray(pixels[s + pick]))
        owners.append(np.full(len(pick), i))
    return np.concatenate(xs).astype(np.float32), np.concatenate(owners)


def fit_band_standardiser(pixels, meta, train_idx, k=200, seed=0):
    """Per-band mean and std over a sample of TRAIN-fold leaf pixels.

    Neural nets train much better when every input has mean 0, std 1. The
    statistics must come from training leaves only; using test leaves would
    leak information about the test set into the model.
    Returns (mean, std), each (204,) float32; std has +1e-6 so no band divides by 0.
    """
    X, _ = sample_pixels_per_leaf(pixels, meta, train_idx, k, seed)
    # why not torch: NumPy's float32 reduction order is what the archived runs used (bit-identical)
    return X.mean(0), X.std(0) + 1e-6


# --------------------------------------------------------------------------
# Saving predictions and computing metrics
# --------------------------------------------------------------------------
def save_oof(name: str, meta: pd.DataFrame, p_rb, p_lr, quick: bool = False,
             log: logging.Logger | None = None) -> Path:
    """Write out-of-fold probabilities for the leaves that were predicted.

    p_rb, p_lr: arrays of length len(meta); NaN for leaves not predicted
    (e.g. when only some folds were run). NaN rows are dropped.
    Writes RESULTS_DIR[/quick]/<name>_oof.csv and returns its path; the
    "[saved]" line goes to `log` if given, else is printed.
    """
    out = meta[["path", "variety", "row", "fold"] + LABELS].copy()
    out["p_rb"], out["p_lr"] = p_rb, p_lr
    out = out.dropna(subset=["p_rb", "p_lr"])
    folder = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_oof.csv"
    out.to_csv(path, index=False)
    _report(f"[saved] {path}  ({len(out)} leaves)", log)
    return path


def _auc(y, s):
    """Fast rank-based AUC (Mann-Whitney U); NaN if only one class is present.

    ~50x faster than sklearn.metrics.roc_auc_score for small arrays, which matters
    because bootstraps (bootstrap_ci, the archived study's compare script) compute
    hundreds of thousands of AUCs. Gives the same value as roc_auc_score.
    y: (n,) 0/1 labels; s: (n,) scores (any monotone scale). Returns a float in [0, 1].
    """
    y = np.asarray(y)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # why not torch: torch has no average-rank-for-ties; scipy's rankdata gives the exact reported AUCs
    ranks = rankdata(s)                      # ties get the average rank
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def p_any_disease(p_rb, p_lr):
    """P(leaf has at least one disease), assuming the two heads are independent.

    p_rb, p_lr: (n,) probabilities -> (n,) probability = 1 - P(no RB) * P(no LR)."""
    return 1.0 - (1.0 - np.asarray(p_rb)) * (1.0 - np.asarray(p_lr))


def _stratified_auc(y, s, strata):
    """AUC computed INSIDE each stratum, then combined (weighted by #pos x #neg pairs).

    This is the usual 'stratified Mann-Whitney' AUC: only leaves in the same
    stratum are ever compared with each other, so differences between strata
    (varieties, or folds whose models are calibrated a little differently) cannot
    create or hide signal. Returns NaN if no stratum has both classes.
    Why it matters here: disease rates differ hugely between varieties (96% of
    Cab Franc leaves have red blotch), so a pooled AUC rewards recognising the
    variety; within-variety AUC only rewards telling sick from healthy leaves
    of the same variety. Strata with one class are skipped.
    y: (n,) 0/1; s: (n,) scores; strata: (n,) labels, e.g. "Baco|3" (variety|fold).
    """
    y, s, strata = np.asarray(y), np.asarray(s), np.asarray(strata)
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        npos, nneg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        num += _auc(y[m], s[m]) * npos * nneg
        den += npos * nneg
    return num / den if den else np.nan


def auc_metrics(y_rb, y_lr, s_rb, s_lr, ok_lr, strata) -> dict:
    """The six AUC metrics from plain arrays (fast; used inside bootstraps).

    y_rb, y_lr : (n,) 0/1 labels; s_rb, s_lr : (n,) predicted probabilities
    ok_lr : bool array, False for leaves excluded from leafroll scoring (Cab Franc)
    strata: array of labels, one per leaf, e.g. "Baco|3" (variety|fold)
    Returns {auc_rb, auc_lr, auc_any, auc_rb_within, auc_lr_within, auc_any_within}.
    """
    y_any = ((y_rb == 1) | (y_lr == 1)).astype(int)
    s_any = p_any_disease(s_rb, s_lr)
    return {
        "auc_rb": _auc(y_rb, s_rb),
        "auc_lr": _auc(y_lr[ok_lr], s_lr[ok_lr]),
        "auc_any": _auc(y_any, s_any),
        "auc_rb_within": _stratified_auc(y_rb, s_rb, strata),
        "auc_lr_within": _stratified_auc(y_lr[ok_lr], s_lr[ok_lr], strata[ok_lr]),
        "auc_any_within": _stratified_auc(y_any, s_any, strata),
    }


def oof_arrays(oof: pd.DataFrame):
    """Turn an OOF table into the arrays auc_metrics() needs:
    (y_rb, y_lr, s_rb, s_lr, ok_lr, strata), each of length len(oof)."""
    return (oof.red_blotch.to_numpy(), oof.leafroll3.to_numpy(),
            oof.p_rb.to_numpy(), oof.p_lr.to_numpy(),
            (oof.variety != NO_LEAFROLL_VARIETY).to_numpy(),
            (oof.variety.astype(str) + "|" + oof.fold.astype(str)).to_numpy())


def evaluate(oof: pd.DataFrame) -> dict:
    """All headline metrics for one approach from its OOF table.

    auc_*        : POOLED AUC over all leaves together. Can look good simply
                   because the model recognises the VARIETY (e.g. 96% of Cab
                   Franc leaves have red blotch).
    auc_*_within : STRATIFIED AUC - leaves are only compared with leaves of the
                   same variety AND the same CV fold. This is the honest number:
                   can the model separate sick from healthy leaves of the same
                   variety? A model that only knows the variety scores 0.50.
    bacc_*       : balanced accuracy at probability threshold 0.5.
    per_variety  : DataFrame, one row per variety, AUCs within the variety
                   (stratified by fold only).
    oof needs columns variety, fold, red_blotch, leafroll3, p_rb, p_lr.
    """
    y_rb, y_lr, s_rb, s_lr, ok_lr, strata = oof_arrays(oof)
    y_any = ((y_rb == 1) | (y_lr == 1)).astype(int)
    s_any = p_any_disease(s_rb, s_lr)
    res = {"n_leaves": len(oof), **auc_metrics(y_rb, y_lr, s_rb, s_lr, ok_lr, strata)}
    res.update({
        "bacc_rb": balanced_accuracy_score(y_rb, s_rb > 0.5),
        "bacc_lr": balanced_accuracy_score(y_lr[ok_lr], s_lr[ok_lr] > 0.5),
        "bacc_any": balanced_accuracy_score(y_any, s_any > 0.5),
    })
    per_variety = []
    for v, g in oof.groupby("variety"):
        yr, yl = g.red_blotch.to_numpy(), g.leafroll3.to_numpy()
        ya = ((yr == 1) | (yl == 1)).astype(int)
        st = g.fold.to_numpy()
        per_variety.append({
            "variety": v,
            "auc_rb": _stratified_auc(yr, g.p_rb, st),
            "auc_lr": _stratified_auc(yl, g.p_lr, st) if v != NO_LEAFROLL_VARIETY else np.nan,
            "auc_any": _stratified_auc(ya, p_any_disease(g.p_rb, g.p_lr), st),
        })
    res["per_variety"] = pd.DataFrame(per_variety)
    return res


def bootstrap_ci(oof: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict:
    """95% bootstrap intervals (resampling leaves) for all six AUC metrics.

    Returns for each metric k: k+"_ci" (half-width), k+"_lo" and k+"_hi" (the 2.5% and
    97.5% percentiles). The "_within" metrics are the ones to compare with 0.5: if the
    interval does not contain 0.5 there is evidence of signal beyond the variety.
    """
    arrays = oof_arrays(oof)
    # why not torch: reported intervals depend on this exact NumPy random stream
    rng = np.random.default_rng(seed)
    vals = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(oof), len(oof))
        for k, v in auc_metrics(*(a[idx] for a in arrays)).items():
            vals.setdefault(k, []).append(v)
    out = {}
    for k, v in vals.items():
        lo, hi = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        out.update({k + "_ci": (hi - lo) / 2, k + "_lo": lo, k + "_hi": hi})
    return out


# --------------------------------------------------------------------------
# Reporting helpers (the only functions here with output side effects)
# --------------------------------------------------------------------------
def print_fold_summary(name, fold, meta, test_idx, p_rb, p_lr, log: logging.Logger | None = None):
    """REPORTING: one progress line per fold (pooled test-fold AUCs, leafroll without Cab Franc).

    p_rb, p_lr: predictions for the leaves in test_idx (same order). The line goes
    to `log` if given, else is printed (the historic behaviour). Returns None.
    """
    sub = meta.iloc[test_idx]
    y_rb = sub.red_blotch.to_numpy()
    y_lr = sub.leafroll3.to_numpy()
    ok = (sub.variety != NO_LEAFROLL_VARIETY).to_numpy()
    _report(f"[{name}] fold {fold}: n={len(test_idx)}  "
            f"AUC red_blotch={_auc(y_rb, p_rb):.3f}  "
            f"leafroll={_auc(y_lr[ok], np.asarray(p_lr)[ok]):.3f}", log)


def write_json(path: Path, obj) -> None:
    """Write obj to path as indented JSON (I/O helper, no printing)."""
    path.write_text(json.dumps(obj, indent=2))
