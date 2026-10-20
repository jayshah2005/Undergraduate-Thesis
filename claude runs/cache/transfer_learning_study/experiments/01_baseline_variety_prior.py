"""BASELINE - "just guess from the variety". Not a real model; a sanity bar.

Run:  python experiments/01_baseline_variety_prior.py

For each test leaf we predict the disease rate of its VARIETY in the training
folds (e.g. ~96% of Cab Franc leaves have red blotch, so every Cab Franc leaf
gets p_rb = 0.96) and look at nothing about the leaf itself.

Why we need it: the labels are very unevenly spread across varieties, so a
model that has only learned to recognise the variety already gets a decent
POOLED AUC. Any real approach must beat this number, and - more importantly -
must beat 0.5 in the "within-variety" AUC (this baseline scores exactly 0.5
there because it gives every leaf of a variety the same score).
"""
import numpy as np

from common import (LABELS, N_FOLDS, fold_indices, label_mask, load_meta,
                    print_fold_summary, save_oof)


def main():
    meta = load_meta()
    y = meta[LABELS].to_numpy()
    mask = label_mask(meta)
    p_rb = np.full(len(meta), np.nan)
    p_lr = np.full(len(meta), np.nan)

    for fold in range(N_FOLDS):
        train_idx, test_idx = fold_indices(meta, fold)
        train = meta.iloc[train_idx]
        for j, out in enumerate((p_rb, p_lr)):
            usable = mask[train_idx, j]
            # mean label per variety using TRAIN leaves only
            prev = (train[usable].groupby("variety")[LABELS[j]].mean())
            out[test_idx] = meta.variety.iloc[test_idx].map(prev).fillna(
                y[train_idx][usable, j].mean()).to_numpy()
        print_fold_summary("01_variety_prior", fold, meta, test_idx,
                           p_rb[test_idx], p_lr[test_idx])

    save_oof("01_variety_prior", meta, p_rb, p_lr)


if __name__ == "__main__":
    main()
