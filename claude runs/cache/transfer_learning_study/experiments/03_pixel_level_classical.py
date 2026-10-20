"""APPROACH 2 - classical models trained on INDIVIDUAL PIXELS, averaged per leaf.

Run:   python experiments/03_pixel_level_classical.py          (full, ~5 min on 14 cores)
       python experiments/03_pixel_level_classical.py --quick  (fold 0, few pixels)

Research basis: reports/Hyperspectral grapevine transfer learning.md - the
user asked that all pixels and all 204 bands be used instead of summaries.
Pixel-level training with leaf-level aggregation is the standard way to do that
(a 2026 tobacco study split by LEAF first and then sampled pixels).

WHAT THIS DOES
  * Every pixel is one training example with 204 features; its label is the
    label of the leaf it came from.
  * To keep it fast we use a random sample of PIXELS_PER_LEAF pixels from each
    TRAINING leaf (default 300 -> ~240k pixels).
  * Models: logistic regression (linear) and histogram gradient boosting (trees).
  * At test time we predict EVERY pixel of each test leaf and AVERAGE the
    probabilities -> one probability per leaf.

TWO THINGS TO KEEP IN MIND
  1. The folds are split by LEAF (see common.py), never by pixel. Pixels of one
     leaf are almost copies of each other; if some were in training and others
     in testing the score would be fake-high.
  2. A pixel from a diseased leaf can look healthy (symptoms are patchy), so the
     pixel labels are noisy. Averaging over the leaf reduces that noise.

NO TORCH HERE - plain scikit-learn.
"""
import os

# Speed: folds/models run in PARALLEL processes, so give each fit only 2 CPU threads.
# Must be set BEFORE sklearn/numpy are imported.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (LABELS, N_FOLDS, fold_indices, label_mask, leaf_pixels,
                    load_meta, load_pixels, print_fold_summary, sample_pixels_per_leaf,
                    save_oof)


def make_model(name):
    if name == "logreg":
        return LogisticRegression(C=0.01, class_weight="balanced", max_iter=500)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                              max_depth=4, class_weight="balanced",
                                              random_state=0)
    raise ValueError(name)


def predict_leaf_mean(model, scaler, pixels, meta, leaf_idx):
    """Average the pixel-level probability over every pixel of each leaf."""
    out = np.zeros(len(leaf_idx))
    for k, i in enumerate(leaf_idx):
        px = scaler.transform(np.asarray(leaf_pixels(pixels, meta, i)))
        out[k] = model.predict_proba(px)[:, 1].mean()
    return out


def run_fold(model_name, fold, per_leaf):
    """One (model, fold) task; runs in its own worker process.

    Each worker opens the pixel file with mmap=True: the operating system keeps ONE
    shared copy of the 6 GB in its page cache instead of every process loading its own.
    Returns (test_idx, (n_test, 2) probabilities).
    """
    meta, pixels = load_meta(), load_pixels(mmap=True)
    y = meta[LABELS].to_numpy()
    mask = label_mask(meta)
    train_idx, test_idx = fold_indices(meta, fold)
    # 1) sample pixels from TRAIN leaves only
    X, owner = sample_pixels_per_leaf(pixels, meta, train_idx, per_leaf, seed=fold)
    # 2) per-band standardisation fitted on those training pixels only
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    out = np.zeros((len(test_idx), 2))
    for j in range(2):
        keep = mask[owner, j]                     # leafroll: skip Cab Franc pixels
        model = make_model(model_name).fit(Xs[keep], y[owner[keep], j])
        out[:, j] = predict_leaf_mean(model, scaler, pixels, meta, test_idx)
    return test_idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="logreg,hgb")
    ap.add_argument("--pixels-per-leaf", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="parallel worker processes (default: half the CPU cores)")
    args = ap.parse_args()

    meta = load_meta()
    folds = [0] if args.quick else list(range(N_FOLDS))
    per_leaf = 30 if args.quick else args.pixels_per_leaf
    models = args.models.split(",")
    tasks = [(m, f) for m in models for f in folds]
    t0 = time.time()
    outs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_fold)(m, f, per_leaf) for m, f in tasks)
    print(f"all fits finished in {time.time() - t0:.0f}s")

    for m in models:
        name = f"03_pixel_{m}"
        p = np.full((len(meta), 2), np.nan)
        for (mm, f), (test_idx, out) in zip(tasks, outs):
            if mm == m:
                p[test_idx] = out
                print_fold_summary(name, f, meta, test_idx, out[:, 0], out[:, 1])
        save_oof(name, meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
