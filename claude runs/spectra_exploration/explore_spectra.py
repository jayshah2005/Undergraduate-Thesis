"""Do the leaf spectra separate by variety or by disease?

Exploration only: nothing is trained except one small cross-validated linear check at the end.

Run from claude runs/spectra_exploration/:
    ../../.venv/bin/python -u explore_spectra.py

Leaf spectrum = the mean over all of the leaf's pixels, all 204 bands (397-1004 nm), read from
the pixel cache by ../shared/spectra.py (cached in ../shared/cache/leaf_mean_spectra.npz; also
written here as results/leaf_mean_spectra.csv). Disease class = the 4 classes of the models:
healthy / RB only / LR only / both. One leaf per vine, so leaves are independent at the vine level.

Every analysis is done twice: on the raw mean spectra, and on SNV-normalised spectra (each leaf
centred and scaled over its own bands), which removes overall brightness (illumination, leaf
angle, imaging session) and keeps the shape. PCA is covariance PCA (bands are not z-scored,
which would inflate the noisy edge bands).

Outputs
  figures/01_spectra_by_variety.png            every leaf coloured by variety, variety means (raw | SNV)
  figures/02_spectra_by_class_per_variety.png  class mean +- 1 SD inside each variety
  figures/03_difference_spectra.png            variety - grand mean | class - healthy (Baco, Chardonnay), one y-axis
  figures/04_pca_all_leaves.png                PCA of all leaves, by variety | by class (PC1-2, PC2-3)
  figures/05_pca_within_baco_chardonnay.png    PCA fitted inside Baco and inside Chardonnay, by class
  figures/06_pca_all_leaves_snv.png            figure 4 on SNV spectra
  figures/07_effect_size_baco_chardonnay.png   per-band Cohen's d: healthy vs RB only / LR only / any disease
  results/results.json, results/tables.md      the numbers
  results/leaf_mean_spectra.csv                one row per leaf: labels, class, glint share, 204 bands
  logs/run.log                                 what the run printed (tables.md + the saved paths)

Terms
  R^2 / eta^2  share of the total sum of squares (squared distances of the leaves from the mean
               spectrum, summed over bands) that lies between the group means. 0 = groups identical.
  chance       the R^2 that randomly shuffled labels give on average: random groups never have
               exactly equal means, so a small R^2 is expected even with no effect.
  Cohen's d    per band: difference of two group means / pooled leaf-to-leaf SD (0.5 medium, 0.8 large).
  row          a vineyard row R1-R5 = one fold of the models. Neighbouring vines look alike, so whole
               rows are held out together; a random split would leak and inflate the scores.
  within-row AUC  AUC that only compares leaves of the same row (../shared/common.py
               _stratified_auc), so a row or session difference cannot create a disease signal.

Numbers
  * Share of the total sum of squares (centred spectra, all bands) explained by variety, by class
    pooled over varieties, by class within variety and by row within variety. Disease prevalence
    differs by row, so the last two overlap and are not additive.
  * First 10 PCs of all leaves: explained variance, eta^2 (between-group / total SS of the
    scores) of variety and of class within variety, and the correlation with leaf brightness.
  * Baco and Chardonnay separately: R^2 (between-group / total SS, all bands) of class, healthy vs
    any disease, the RB and LR labels, and row, with the R^2 expected by chance (permutation mean).
    Disease p-values permute labels WITHIN rows, so a row (or imaging-session) effect cannot pass
    for disease.
  * Leave-one-row-out logistic regression on the 204-band spectra (rows = the models' folds;
    scaler fitted inside each fold, C = 1, not tuned): variety balanced accuracy over all leaves,
    and inside Baco / Chardonnay the AUC of healthy vs any disease, RB and LR computed within rows
    (= the models' within-variety AUC).
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from sklearn.decomposition import PCA                                # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, silhouette_score  # noqa: E402
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))                     # common.py, spectra.py, logs.py
from common import _stratified_auc, load_meta, load_wavelengths      # noqa: E402
from logs import get_logger                                          # noqa: E402
from spectra import leaf_mean_spectra, snv                           # noqa: E402

# Why not torch: every number here is already reported (REPORT.md). The p-values come from the
# numpy RNG stream, the AUCs from scikit-learn's LogisticRegression, and PCA / silhouette from
# scikit-learn; swapping libraries would change them, so this file stays numpy / sklearn.

FIG, RES, LOGS = HERE / "figures", HERE / "results", HERE / "logs"
CLASSES = ["healthy", "RB only", "LR only", "both"]
CLASS_COLORS = ["#009E73", "#D55E00", "#56B4E9", "#6A3D9A"]
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
VAR_COLORS = {"Baco": "#E69F00", "Cab Franc": "#CC79A7", "Chardonnay": "#0072B2", "Vidal": "#555555"}
VAR_MARKERS = {"Baco": "o", "Cab Franc": "s", "Chardonnay": "^", "Vidal": "D"}
FOCUS = ["Baco", "Chardonnay"]
N_PCS, N_PERM, SEED = 10, 999, 0


# --------------------------------------------------------------------------
# Sums of squares (computation; no printing or files)
# numpy float64, not torch: the permutation p-values compare each shuffled R^2 with the observed
# one (v >= obs), so even last-digit differences in the summation could flip a tie.
# --------------------------------------------------------------------------
def total_ss(X):
    """Sum of squared differences of every leaf from the mean spectrum, over all bands."""
    return float(((X - X.mean(0)) ** 2).sum())


def between_ss(X, g):
    """Between-group sum of squares over all columns."""
    gm = X.mean(0)
    return float(sum((g == k).sum() * ((X[g == k].mean(0) - gm) ** 2).sum() for k in np.unique(g)))


def between_ss_within(X, g, strata):
    """Between-group SS computed inside each stratum, summed (group effect net of the strata)."""
    return float(sum(between_ss(X[strata == s], g[strata == s]) for s in np.unique(strata)))


def r2(X, g):
    """Share of the total sum of squares lying between the groups of g (0 = none, 1 = all)."""
    return between_ss(X, g) / total_ss(X)


def permute_within(g, strata, rng):
    """Shuffle the labels g separately inside each stratum, keeping each stratum's label counts.

    Why within rows: disease prevalence differs by row and rows differ spectrally (imaging session,
    position), so a free shuffle would let a row effect pass for disease. Shuffling inside each row
    keeps that row effect in the null distribution. numpy rng: its stream gives the reported p."""
    out = g.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = rng.permutation(g[idx])
    return out


def r2_test(X, g, strata, rng, n_perm=N_PERM):
    """R^2 of g, a permutation p-value and the R^2 expected by chance (mean of the null);
    strata=None permutes freely, otherwise within strata.

    p = (1 + #shuffles with R^2 >= observed) / (1 + n_perm): the share of label shufflings that
    separate the leaves at least as well as the real labels (the +1 counts the real labels)."""
    obs = r2(X, g)
    null = [r2(X, rng.permutation(g) if strata is None else permute_within(g, strata, rng)) for _ in range(n_perm)]
    return obs, (1 + sum(v >= obs for v in null)) / (1 + n_perm), float(np.mean(null))


def cohens_d(a, b):
    """Per band: (mean b - mean a) / pooled SD, i.e. the group difference in units of the
    leaf-to-leaf spread. a, b: (n_a, 204), (n_b, 204) -> (204,). numpy: the peaks go into results.json."""
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean(0) - a.mean(0)) / sp


# --------------------------------------------------------------------------
# Leave-one-row-out linear check
# --------------------------------------------------------------------------
def loro_proba(X, y, rows):
    """Out-of-row predicted probabilities of a scaled logistic regression (one model per held-out row).

    X (n, 204), y (n,) labels, rows (n,) -> P (n, n_classes). Whole rows are held out because
    neighbouring vines look alike (a random split would leak); the scaler sits in the pipeline so it
    is fitted on the training rows only. sklearn LogisticRegression, not torch: it is the reported model."""
    P = np.empty((len(y), len(np.unique(y))))
    for r in np.unique(rows):
        te = rows == r
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        model.fit(X[~te], y[~te])
        P[te] = model.predict_proba(X[te])
    return P


# --------------------------------------------------------------------------
# Analysis of one representation (raw or SNV)
# --------------------------------------------------------------------------
def analyse(X, meta, brightness, rng) -> dict:
    """All numbers for one representation (raw or SNV): variance split, PCA of all leaves, silhouette,
    the within-Baco / Chardonnay tests and the leave-one-row-out variety classifier.

    X (998, 204) spectra, brightness (998,) raw mean reflectance per leaf, rng the shared numpy
    generator (the call order below fixes which random numbers each test gets - do not reorder)."""
    var, cls, row = meta.variety.to_numpy(), meta.cls.to_numpy(), meta.row.to_numpy()
    # 1. variance split over all leaves
    tot = total_ss(X)
    out = {"variance_share": {
        "variety": between_ss(X, var) / tot,
        "class (pooled over varieties)": between_ss(X, cls) / tot,
        "class within variety": between_ss_within(X, cls, var) / tot,
        "row within variety": between_ss_within(X, row, var) / tot,
        "variety x row cells": between_ss(X, np.char.add(var.astype(str), row.astype(str))) / tot}}

    # 2. PCA of all leaves (sklearn PCA and silhouette, not torch: reported values)
    pca = PCA(N_PCS).fit(X)
    Z = pca.transform(X)
    out["pca_all"] =[{"pc": k + 1, "explained": float(pca.explained_variance_ratio_[k]),
                       "eta2 variety": r2(Z[:, [k]], var),
                       "eta2 class within variety": between_ss_within(Z[:, [k]], cls, var) / total_ss(Z[:, [k]]),
                       "corr with brightness": float(np.corrcoef(Z[:, k], brightness)[0, 1])}
                      for k in range(N_PCS)]
    out["silhouette_all_pc1-5"] = {"variety": float(silhouette_score(Z[:, :5], var)),
                                   "class": float(silhouette_score(Z[:, :5], cls))}

    # 3. inside Baco and inside Chardonnay: does disease separate the leaves? (one variety at a time,
    #    so the large variety difference cannot count as disease)
    out["within"] = {}
    for v in FOCUS:
        m = var == v
        Xv, cv, rv = X[m], cls[m], row[m]
        targets = {"class (4)": cv, "healthy vs any disease": (cv > 0).astype(int),
                   "RB label": meta.red_blotch.to_numpy()[m], "LR label": meta.leafroll3.to_numpy()[m]}
        res = {}
        for name, g in targets.items():
            r, p, chance = r2_test(Xv, g, rv, rng)
            res[name] = {"R2": r, "p (permuted within rows)": p, "R2 by chance": chance}
        # row effect for scale; rows are the thing being tested, so they are shuffled freely
        r, p, chance = r2_test(Xv, rv, None, rng)
        res["row"] ={"R2": r, "p (free permutation)": p, "R2 by chance": chance}
        pv = PCA(N_PCS).fit(Xv)
        Zv = pv.transform(Xv)
        res["pca"] = [{"pc": k + 1, "explained": float(pv.explained_variance_ratio_[k]),
                       "eta2 class": r2(Zv[:, [k]], cv), "eta2 row": r2(Zv[:, [k]], rv)} for k in range(N_PCS)]
        res["silhouette_pc1-5"] = {"class": float(silhouette_score(Zv[:, :5], cv)),
                                   "row": float(silhouette_score(Zv[:, :5], rv))}
        # within-row AUC of the binary targets (the 4-class target has no single AUC)
        res["auc_loro_within_rows"] = {
            name: float(_stratified_auc(g, loro_proba(Xv, g, rv)[:, 1], rv))
            for name, g in list(targets.items())[1:]}
        out["within"][v] = res

    # 4. how well the variety itself is predicted (balanced accuracy: mean recall over the 4 varieties)
    P = loro_proba(X, var, row)
    out["variety_loro_balanced_accuracy"] = float(balanced_accuracy_score(var, np.array(sorted(set(var)))[P.argmax(1)]))
    return out


# --------------------------------------------------------------------------
# Per-band effect sizes (computation)
# --------------------------------------------------------------------------
# (name, which classes count as the diseased group, plot colour); the healthy group is class 0
DISEASE_CONTRASTS = [("RB only", lambda c: c == 1, CLASS_COLORS[1]), ("LR only", lambda c: c == 2, CLASS_COLORS[2]),
                     ("any disease", lambda c: c > 0, "k")]
VARIETY_PAIRS = [("Baco", "Chardonnay"), ("Baco", "Vidal"), ("Chardonnay", "Vidal"), ("Cab Franc", "Vidal")]


def disease_effect_curves(X, meta) -> dict:
    """Per-band Cohen's d of healthy vs each disease contrast inside Baco and Chardonnay (raw spectra).
    Returns {variety: {contrast name: (204,) d}}, used by figure 07 and by effect_size_peaks."""
    cls, var = meta.cls.to_numpy(), meta.variety.to_numpy()
    return {v: {name: cohens_d(X[(var == v) & (cls == 0)], X[(var == v) & sel(cls)])
                for name, sel, _ in DISEASE_CONTRASTS} for v in FOCUS}


def effect_size_peaks(w, X, meta, curves) -> dict:
    """Largest |d| and its wavelength for every disease contrast, plus four variety pairs for scale."""
    def peak(d):
        k = int(np.abs(d).argmax())
        return {"max |d|": float(abs(d[k])), "at nm": float(w[k])}
    peaks = {f"{v}: healthy vs {name}": peak(d) for v, ds in curves.items() for name, d in ds.items()}
    var = meta.variety.to_numpy()
    for a, b in VARIETY_PAIRS:
        peaks[f"variety: {a} vs {b}"] = peak(cohens_d(X[var == a], X[var == b]))
    return peaks


# --------------------------------------------------------------------------
# Reporting: figures (PCA is refitted here for plotting only; the reported PCA numbers come from analyse)
# --------------------------------------------------------------------------
def fig_spectra_by_variety(w, X, Xs, meta):
    """Figure 01: every leaf's spectrum coloured by variety, with the variety means (raw | SNV)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, A, title, ylab in [(axes[0], X, "Raw mean spectra", "reflectance"),
                               (axes[1], Xs, "SNV-normalised (brightness removed)", "SNV value")]:
        for v in VARIETIES:
            m = meta.variety.to_numpy() == v
            ax.plot(w, A[m].T, color=VAR_COLORS[v], lw=0.4, alpha=0.12)
        for v in VARIETIES:
            m = meta.variety.to_numpy() == v
            ax.plot(w, A[m].mean(0), color=VAR_COLORS[v], lw=2.5, label=f"{v} (n={m.sum()})",
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground="white")])
        ax.set(title=title, xlabel="wavelength (nm)", ylabel=ylab, xlim=(w[0], w[-1]))
        ax.legend(loc="upper left")
    fig.suptitle("One mean spectrum per leaf (thin) and variety means (bold)")
    fig.tight_layout()
    fig.savefig(FIG / "01_spectra_by_variety.png", dpi=150)
    plt.close(fig)


def fig_spectra_by_class(w, X, meta):
    """Figure 02: disease-class mean +- 1 SD inside each variety (raw)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, VARIETIES):
        mv = meta.variety.to_numpy() == v
        for c, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
            m = mv & (meta.cls.to_numpy() == c)
            if not m.any():
                continue
            mu, sd = X[m].mean(0), X[m].std(0)
            ax.fill_between(w, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.plot(w, mu, color=col, lw=2, label=f"{name} (n={m.sum()})")
        ax.set(title=v, xlim=(w[0], w[-1]))
        ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("reflectance")
    fig.suptitle("Disease-class mean spectra within each variety (shading = +- 1 SD over leaves)")
    fig.tight_layout()
    fig.savefig(FIG / "02_spectra_by_class_per_variety.png", dpi=150)
    plt.close(fig)


def fig_differences(w, X, meta):
    """Figure 03: variety mean - grand mean next to class mean - healthy mean (Baco, Chardonnay)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    gm = X.mean(0)
    for v in VARIETIES:
        m = meta.variety.to_numpy() == v
        axes[0].plot(w, X[m].mean(0) - gm, color=VAR_COLORS[v], lw=2, label=v)
    axes[0].set(title="Variety mean - grand mean")
    for ax, v in zip(axes[1:], FOCUS):
        mv = meta.variety.to_numpy() == v
        healthy = X[mv & (meta.cls.to_numpy() == 0)].mean(0)
        for c in (1, 2, 3):
            m = mv & (meta.cls.to_numpy() == c)
            ax.plot(w, X[m].mean(0) - healthy, color=CLASS_COLORS[c], lw=2, label=f"{CLASSES[c]} (n={m.sum()})")
        ax.set(title=f"{v}: class mean - healthy mean (n={(mv & (meta.cls.to_numpy() == 0)).sum()})")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel="wavelength (nm)", xlim=(w[0], w[-1]))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("reflectance difference")
    fig.suptitle("How big is the variety difference compared with the disease difference? (same y-axis)")
    fig.tight_layout()
    fig.savefig(FIG / "03_difference_spectra.png", dpi=150)
    plt.close(fig)


def scatter_by(ax, Z, i, j, meta, by, ev):
    """PC i vs PC j, coloured by variety, or by class with the variety as the marker."""
    if by == "variety":
        for v in VARIETIES:
            m = meta.variety.to_numpy() == v
            ax.scatter(Z[m, i], Z[m, j], s=10, alpha=0.6, color=VAR_COLORS[v], marker=VAR_MARKERS[v], label=v, lw=0)
    else:
        for c, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
            for v in VARIETIES:
                m = (meta.cls.to_numpy() == c) & (meta.variety.to_numpy() == v)
                ax.scatter(Z[m, i], Z[m, j], s=10, alpha=0.6, color=col, marker=VAR_MARKERS[v], lw=0,
                           label=name if v == VARIETIES[0] else None)
    ax.set(xlabel=f"PC{i + 1} ({ev[i]:.1%})", ylabel=f"PC{j + 1} ({ev[j]:.1%})")
    ax.legend(loc="best", markerscale=2, fontsize=8)


def fig_pca_all(X, meta, path, title):
    """Figures 04 / 06: PCA of all leaves (PC1-2, PC2-3), by variety and by class."""
    pca = PCA(N_PCS).fit(X)
    Z, ev = pca.transform(X), pca.explained_variance_ratio_
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for r, (i, j) in enumerate([(0, 1), (1, 2)]):
        scatter_by(axes[r, 0], Z, i, j, meta, "variety", ev)
        scatter_by(axes[r, 1], Z, i, j, meta, "class", ev)
    axes[0, 0].set_title("coloured by variety")
    axes[0, 1].set_title("coloured by disease class (marker = variety)")
    fig.text(0.5, 0.005, "Pooled class colouring partly repeats variety: Cab Franc is 96% RB only, Vidal 63%.",
             ha="center", fontsize=9, style="italic")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pca_within(X, meta):
    """Figure 05: PCA fitted inside Baco and inside Chardonnay, coloured by class."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for r, v in enumerate(FOCUS):
        m = meta.variety.to_numpy() == v
        pca = PCA(N_PCS).fit(X[m])
        Z, ev, cv = pca.transform(X[m]), pca.explained_variance_ratio_, meta.cls.to_numpy()[m]
        for col, (i, j) in enumerate([(0, 1), (2, 3)]):
            ax = axes[r, col]
            for c, (name, colr) in enumerate(zip(CLASSES, CLASS_COLORS)):
                ax.scatter(Z[cv == c, i], Z[cv == c, j], s=14, alpha=0.7, color=colr, lw=0,
                           label=f"{name} (n={(cv == c).sum()})")
            ax.set(title=f"{v} only: PC{i + 1} vs PC{j + 1}", xlabel=f"PC{i + 1} ({ev[i]:.1%})",
                   ylabel=f"PC{j + 1} ({ev[j]:.1%})")
            ax.legend(loc="best", markerscale=1.5, fontsize=8)
    fig.suptitle("PCA fitted inside one variety (raw mean spectra), coloured by disease class")
    fig.tight_layout()
    fig.savefig(FIG / "05_pca_within_baco_chardonnay.png", dpi=150)
    plt.close(fig)


def fig_effect_sizes(w, curves):
    """Figure 07: per-band Cohen's d of disease inside Baco and Chardonnay (curves from disease_effect_curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, v in zip(axes, FOCUS):
        for name, _, colr in DISEASE_CONTRASTS:
            ax.plot(w, curves[v][name], color=colr, lw=2, label=f"healthy vs {name}")
        for y in (-0.5, 0.5):
            ax.axhline(y, color="grey", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(title=v, xlabel="wavelength (nm)", xlim=(w[0], w[-1]))
        ax.legend(loc="best")
    axes[0].set_ylabel("Cohen's d (diseased - healthy)")
    fig.suptitle("Per-band effect size of disease within variety (dotted = |d| 0.5, a 'medium' effect)")
    fig.tight_layout()
    fig.savefig(FIG / "07_effect_size_baco_chardonnay.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporting: tables and result files
# --------------------------------------------------------------------------
def f3(x):
    """Number with 3 decimals."""
    return f"{x:.3f}"


def tables(res: dict, glint: np.ndarray) -> str:
    """results/tables.md: the numbers of both representations as markdown tables."""
    raw, sn = res["raw"], res["snv"]
    L = ["# Leaf spectra: variety vs disease - numbers", "",
         f"Specular glints: {glint.mean():.2%} of a leaf's pixels have a band above 1.0 on average "
         f"(max {glint.max():.1%}); they are included in the means.", "",
         "## Share of the total variance (sum of squares, all bands) explained by", "",
         "| grouping | raw | SNV |", "|---|---|---|"]
    L += [f"| {k} | {f3(raw['variance_share'][k])} | {f3(sn['variance_share'][k])} |" for k in raw["variance_share"]]
    L += ["", "Class within variety and row within variety overlap (disease prevalence differs by row).", ""]
    for name, r in (("raw", raw), ("SNV", sn)):
        L += [f"## PCA of all leaves ({name}): first {N_PCS} PCs", "",
              "| PC | explained | eta2 variety | eta2 class within variety | corr with brightness |", "|---|---|---|---|---|"]
        L += [f"| {p['pc']} | {p['explained']:.1%} | {f3(p['eta2 variety'])} | {f3(p['eta2 class within variety'])} | "
              f"{p['corr with brightness']:+.2f} |" for p in r["pca_all"]]
        s = r["silhouette_all_pc1-5"]
        L += ["", f"Silhouette in PC1-5 (1 = tight separate clusters, 0 = none): by variety {f3(s['variety'])}, "
              f"by class {f3(s['class'])}. Leave-one-row-out logistic regression predicts the variety with "
              f"balanced accuracy {f3(r['variety_loro_balanced_accuracy'])} (chance 0.25).", ""]
    for v in FOCUS:
        L += [f"## {v}: does disease separate the leaves?", "",
              "R2 = share of this variety's spectral variance between the groups; chance = the R2 that shuffled "
              "labels give on average. p permutes the labels within rows (999 permutations). AUC = "
              "leave-one-row-out logistic regression, computed within rows (0.5 = chance).", "",
              "| target | R2 raw | chance raw | p raw | AUC raw | R2 SNV | chance SNV | p SNV | AUC SNV |",
              "|---|---|---|---|---|---|---|---|---|"]
        a, b = raw["within"][v], sn["within"][v]
        for t in ("class (4)", "healthy vs any disease", "RB label", "LR label"):
            auc_a, auc_b = a["auc_loro_within_rows"].get(t), b["auc_loro_within_rows"].get(t)
            L.append(f"| {t} | {f3(a[t]['R2'])} | {f3(a[t]['R2 by chance'])} | {a[t]['p (permuted within rows)']:.3f} | "
                     f"{f3(auc_a) if auc_a is not None else '-'} | {f3(b[t]['R2'])} | {f3(b[t]['R2 by chance'])} | "
                     f"{b[t]['p (permuted within rows)']:.3f} | {f3(auc_b) if auc_b is not None else '-'} |")
        L.append(f"| row (for scale; p from free permutation) | {f3(a['row']['R2'])} | {f3(a['row']['R2 by chance'])} | "
                 f"{a['row']['p (free permutation)']:.3f} | | {f3(b['row']['R2'])} | {f3(b['row']['R2 by chance'])} | "
                 f"{b['row']['p (free permutation)']:.3f} | |")
        L += ["", f"PCA inside {v} (raw): eta2 of class / row per PC: "
              + ", ".join(f"PC{p['pc']} {p['eta2 class']:.2f}/{p['eta2 row']:.2f}" for p in a["pca"])
              + f". Silhouette PC1-5 by class {f3(a['silhouette_pc1-5']['class'])}, by row {f3(a['silhouette_pc1-5']['row'])}.", ""]
    L += ["## Largest per-band effect sizes (Cohen's d, raw spectra)", "", "| comparison | max abs d | at nm |", "|---|---|---|"]
    L += [f"| {k} | {v['max |d|']:.2f} | {v['at nm']:.0f} |" for k, v in res["effect_size_peaks"].items()]
    return "\n".join(L) + "\n"


def leaf_table(meta, w, X, glint) -> pd.DataFrame:
    """results/leaf_mean_spectra.csv: one row per leaf with its labels, class, glint share and 204 bands."""
    table = meta[["path", "variety", "row", "vine", "red_blotch", "leafroll3"]].copy()
    table["class"] = [CLASSES[c] for c in meta.cls]
    table["glint_pixel_share"] = glint
    return pd.concat([table, pd.DataFrame(X, columns=[f"{x:.1f}nm" for x in w])], axis=1)


def write_figures(w, X, Xs, meta, curves):
    """figures/01-07."""
    fig_spectra_by_variety(w, X, Xs, meta)
    fig_spectra_by_class(w, X, meta)
    fig_differences(w, X, meta)
    fig_pca_all(X, meta, FIG / "04_pca_all_leaves.png", "PCA of the leaf mean spectra (raw, all 998 leaves)")
    fig_pca_within(X, meta)
    fig_pca_all(Xs, meta, FIG / "06_pca_all_leaves_snv.png", "PCA of the SNV-normalised leaf mean spectra (all 998 leaves)")
    fig_effect_sizes(w, curves)


def write_results(res: dict, glint: np.ndarray, leaves: pd.DataFrame) -> str:
    """results/leaf_mean_spectra.csv, results.json and tables.md. Returns the markdown for the log."""
    leaves.to_csv(RES / "leaf_mean_spectra.csv", index=False)
    (RES / "results.json").write_text(json.dumps(res, indent=2))
    md = tables(res, glint)
    (RES / "tables.md").write_text(md)
    return md


def load_data():
    """Leaf metadata (+ 4-class label cls = RB + 2 * LR), wavelengths (204,) nm and the
    (998, 204) raw leaf mean spectra with each leaf's glint share."""
    meta = load_meta()
    meta["cls"] = meta.red_blotch + 2 * meta.leafroll3
    assert meta.groupby(["variety", "row", "vine"]).size().max() == 1, "expected one leaf per vine"
    X, glint = leaf_mean_spectra(meta)
    return meta, load_wavelengths(), X, glint


def compute(meta, w, X, glint) -> tuple[dict, dict]:
    """Every number of the run (raw first, then SNV, sharing one seeded numpy rng, as published),
    plus the disease Cohen's d curves for figure 07."""
    rng = np.random.default_rng(SEED)                              # numpy stream behind the reported p-values
    brightness = X.mean(1)                                         # raw mean reflectance per leaf
    res = {"n_leaves": len(meta), "raw": analyse(X, meta, brightness, rng),
           "snv": analyse(snv(X), meta, brightness, rng)}
    curves = disease_effect_curves(X, meta)
    res["effect_size_peaks"] = effect_size_peaks(w, X, meta, curves)
    res["glint_pixel_share"] = {"mean": float(glint.mean()), "max": float(glint.max())}
    return res, curves


def main():
    """Load leaf spectra -> analyse raw and SNV -> write figures, result files and logs/run.log."""
    for d in (FIG, RES, LOGS):
        d.mkdir(exist_ok=True)
    (LOGS / "run.log").unlink(missing_ok=True)                     # the log holds the latest run only
    log = get_logger("explore_spectra", log_file=LOGS / "run.log")

    meta, w, X, glint = load_data()
    res, curves = compute(meta, w, X, glint)

    md = write_results(res, glint, leaf_table(meta, w, X, glint))
    write_figures(w, X, snv(X), meta, curves)
    log.info(md)
    log.info(f"[saved] {FIG}/*.png, {RES / 'tables.md'}, {RES / 'results.json'}, {RES / 'leaf_mean_spectra.csv'}")


if __name__ == "__main__":
    main()
