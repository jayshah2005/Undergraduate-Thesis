"""DIAGNOSTICS - how much real spectral signal is in this data, and where?

Run:   python experiments/10_diagnostics.py
Takes ~1-2 minutes, plain scikit-learn / matplotlib (no GPU). Prints a report and
writes results/diagnostics.txt and results/diagnostics_band_auc.png.

WHY: the first model results are only slightly above "just guess from the
variety". Before trusting or tuning any model we want to know whether that is
a bug, a leakage artefact or a genuine property of the data. These checks are
cheap and each answers one question:

  A. How strongly does variety alone predict the labels?  (the shortcut)
  B. Are labels spatially clustered along a row (neighbouring vines alike)?
     If yes, random splits leak and "row" splits are the right choice.
  C. How much does a LEAKY random split inflate the score vs our row folds?
  D. Do trivial leaf properties (size, brightness) carry signal?
  E. Which wavelengths separate sick from healthy WITHIN a variety?
     (univariate AUC per band - the figure shows where the signal lives)
  F. Models trained and tested INSIDE one variety (no variety shortcut possible).

All "within-variety" numbers compare a leaf only with leaves of the same variety.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (CACHE_DIR, LABELS, NO_LEAFROLL_VARIETY, RESULTS_DIR, VARIETIES,
                    _stratified_auc, fold_indices, load_meta, load_wavelengths)

OUT = []
# one fixed colour per variety so both panels (and the legend) agree
VARIETY_COLOURS = {"Baco": "C0", "Cab Franc": "C1", "Chardonnay": "C2", "Vidal": "C3"}


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.append(line)


def cv_scores(X, y, meta, splits, C=0.1):
    """Out-of-fold logistic-regression scores for one label."""
    s = np.full(len(y), np.nan)
    for tr, te in splits:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                               max_iter=3000))
        s[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return s


def main():
    meta = load_meta()
    stats = np.load(CACHE_DIR / "leaf_stats.npy")          # (N, 4, 204): mean/std/p10/p90
    mean_spec = stats[:, 0]
    wl = load_wavelengths()
    variety = meta.variety.to_numpy()
    meta["any_disease"] = ((meta.red_blotch == 1) | (meta.leafroll3 == 1)).astype(int)
    labels = {"red_blotch": meta.red_blotch.to_numpy(), "leafroll3": meta.leafroll3.to_numpy()}
    ok = {"red_blotch": np.ones(len(meta), bool), "leafroll3": variety != NO_LEAFROLL_VARIETY}
    row_splits = [fold_indices(meta, k) for k in range(5)]
    say("=" * 70)

    # ---- A. variety alone ---------------------------------------------------
    say("\nA. Label prevalence per variety (this is the shortcut a model can learn)")
    say(meta.groupby("variety")[LABELS + ["any_disease"]].mean().round(2).to_string())

    # ---- B. spatial clustering of labels along rows -------------------------
    say("\nB. Do neighbouring vines share labels?  AUC of 'mean label of the 2 nearest "
        "vines in the same row' as a predictor (0.5 = no clustering)")
    for lab in LABELS:
        preds, ys, strata = [], [], []
        for (v, r), g in meta.groupby(["variety", "row"]):
            g = g.sort_values("vine")
            vals = g[lab].to_numpy(float)
            for k in range(len(g)):
                nb = [vals[j] for j in (k - 1, k + 1) if 0 <= j < len(g)]
                preds.append(np.mean(nb)); ys.append(vals[k]); strata.append(v)
        keep = np.array(strata) != NO_LEAFROLL_VARIETY if lab == "leafroll3" else np.ones(len(ys), bool)
        a = _stratified_auc(np.array(ys)[keep], np.array(preds)[keep], np.array(strata)[keep])
        say(f"   {lab:11s} neighbour-label AUC within variety = {a:.3f}")

    # ---- C. leaky vs grouped CV ---------------------------------------------
    say("\nC. Logistic regression on mean spectra: row-grouped CV (ours) vs LEAKY random CV")
    strata_v = variety
    for lab in LABELS:
        y, m = labels[lab], ok[lab]
        idx = np.where(m)[0]
        grouped = cv_scores(mean_spec, y, meta, [(tr[m[tr]], te) for tr, te in row_splits])
        leaky_splits = [(idx[a], idx[b]) for a, b in
                        StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx])]
        leaky = cv_scores(mean_spec, y, meta, leaky_splits)
        for nm, s in (("row-grouped", grouped), ("random (leaky)", leaky)):
            say(f"   {lab:11s} {nm:15s} pooled AUC {roc_auc_score(y[m], s[m]):.3f}   "
                f"within-variety AUC {_stratified_auc(y[m], s[m], strata_v[m]):.3f}")

    # ---- D. trivial leaf properties -----------------------------------------
    say("\nD. Trivial properties as single predictors, within-variety AUC "
        "(0.5 = useless; <0.5 means higher value -> LESS disease)")
    props = {"leaf pixel count (area)": meta.n_px.to_numpy(float),
             "mean brightness (all bands)": mean_spec.mean(1),
             "mean NIR/red ratio": mean_spec[:, 150:190].mean(1) / mean_spec[:, 85:95].mean(1)}
    for pname, vals in props.items():
        res = []
        for lab in LABELS:
            m = ok[lab]
            res.append(f"{lab} {_stratified_auc(labels[lab][m], vals[m], variety[m]):.3f}")
        say(f"   {pname:30s} " + "   ".join(res))

    # ---- E. univariate AUC per band, within variety -------------------------
    say("\nE. Best single wavelengths (within-variety univariate AUC; figure saved)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, lab in zip(axes, LABELS):
        y, m = labels[lab], ok[lab]
        band_auc = np.zeros(len(wl))
        for b in range(len(wl)):
            band_auc[b] = _stratified_auc(y[m], mean_spec[m, b], variety[m])
        for v in VARIETIES:
            if v == NO_LEAFROLL_VARIETY and lab == "leafroll3":
                continue
            mv = variety == v
            curve = [roc_auc_score(y[mv], mean_spec[mv, b]) if len(np.unique(y[mv])) == 2 else np.nan
                     for b in range(len(wl))]
            ax.plot(wl, curve, lw=0.9, alpha=0.8, label=v, color=VARIETY_COLOURS[v])
        ax.plot(wl, band_auc, "k", lw=2, label="all (within-variety)")
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{lab}: AUC of single band (mean leaf reflectance)")
        ax.set_xlabel("wavelength (nm)")
        top = np.argsort(-np.abs(band_auc - 0.5))[:5]
        say(f"   {lab:11s} top bands: " + ", ".join(f"{wl[b]:.0f}nm (AUC {band_auc[b]:.3f})" for b in top))
    axes[0].set_ylabel("AUC  (0.5 = no signal)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "diagnostics_band_auc.png", dpi=130)

    # ---- F. within-variety models --------------------------------------------
    say("\nF. Logistic regression TRAINED AND TESTED INSIDE one variety (5-fold by row), "
        "mean spectra, AUC")
    for lab in LABELS:
        parts = []
        for v in VARIETIES:
            mv = (variety == v) & ok[lab]
            if v == NO_LEAFROLL_VARIETY and lab == "leafroll3":
                parts.append(f"{v} -"); continue
            y = labels[lab]
            if len(np.unique(y[mv])) < 2 or min((y[mv] == 0).sum(), (y[mv] == 1).sum()) < 10:
                parts.append(f"{v} n/a (too few of one class)"); continue
            splits = [(np.where(mv & (meta.fold != k))[0], np.where(mv & (meta.fold == k))[0])
                      for k in range(5)]
            s = cv_scores(mean_spec, y, meta, splits)
            parts.append(f"{v} {roc_auc_score(y[mv], s[mv]):.3f}")
        say(f"   {lab:11s} " + "   ".join(parts))
    say("\n" + "=" * 70)
    (RESULTS_DIR / "diagnostics.txt").write_text("\n".join(OUT) + "\n")
    print(f"\nsaved {RESULTS_DIR / 'diagnostics.txt'} and diagnostics_band_auc.png")


if __name__ == "__main__":
    main()
