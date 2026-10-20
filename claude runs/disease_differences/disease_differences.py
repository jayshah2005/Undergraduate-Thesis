"""Are healthy, red blotch (RB) and leafroll (LR) leaves spectrally different - inside each variety?

Run from claude runs/disease_differences/:
    ../../.venv/bin/python -u disease_differences.py

Groups (inside one variety at a time, so variety differences cannot count as disease):
    healthy = neither virus, RB only = red blotch alone, LR only = leafroll 3 alone.
    Leaves with both viruses are left out so each group has one disease. Cab Franc has no
    LR-only leaves and only 9 healthy ones, so only healthy vs RB is possible there.
Leaf spectrum = the mean over all of the leaf's pixels (204 bands, 400-1000 nm). Every test is run
on the raw spectra and after SNV (each leaf centred and scaled over its own bands: brightness
removed, shape kept).

Comparisons per variety: healthy vs RB only, healthy vs LR only, RB only vs LR only, and all three
groups together. For each:
  R2      share of the variety's spectral variance (all bands) that lies between the groups
  chance  the R2 that shuffled labels give on average (random groups never have identical means)
  p       share of 999 shuffles of the group labels, done WITHIN rows R1-R5 (each row keeps its mix
          of groups), with an R2 at least as large as the real one. Within rows, because disease
          prevalence differs by row and rows differ spectrally, so a row effect cannot pass for disease
  p Holm  p corrected for running all these tests (Holm's method); use this one for 'significant'
  d       Cohen's d per band = difference of the group means / pooled leaf-to-leaf SD; the largest
          |d| and its wavelength (0.2 small, 0.5 medium, 0.8 large)
  AUC     leave-one-row-out logistic regression (SNV, standardised, C picked inside each training
          split): the chance the classifier ranks a leaf of the second group above one of the
          first, compared within rows (0.5 = guessing). Only when both groups have >= 10 leaves.
          Rows are held out whole because neighbouring vines look alike (a random split would leak).
          C is picked by an inner leave-one-row-out on the training rows only ("nested" CV): choosing
          it on the test row would let the test row tune the model and make the AUC optimistic.

Outputs
  figures/<variety>.png    per variety: (a) raw group means +- 1 SD, (b) SNV group means +- 1 SD,
                           (c) per-band Cohen's d (SNV) for each pair, (d) PCA of the variety's SNV spectra
  figures/summary_effect_sizes.png   largest |d| per comparison and variety (raw and SNV)
  results/disease_differences.md / .json / .csv    all numbers
  logs/run.log             what the run printed (the markdown table)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from sklearn.decomposition import PCA                                # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import log_loss                                 # noqa: E402
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))                     # common.py, spectra.py, logs.py
from common import _stratified_auc, load_meta, load_wavelengths      # noqa: E402
from logs import get_logger                                          # noqa: E402
from spectra import leaf_mean_spectra, snv                           # noqa: E402

# Why not torch: every number here is reported (README, ../spectra_exploration/REPORT.md). The p-values
# come from the numpy RNG stream and the AUCs from scikit-learn's LogisticRegression; swapping
# libraries would change them, so this file stays numpy / sklearn.

FIG, RES, LOGS = HERE / "figures", HERE / "results", HERE / "logs"
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
GROUPS = {0: "healthy", 1: "RB only", 2: "LR only"}                  # class = red_blotch + 2 * leafroll3
COLORS = {"healthy": "#009E73", "RB only": "#D55E00", "LR only": "#56B4E9"}
PAIRS = [("healthy", "RB only"), ("healthy", "LR only"), ("RB only", "LR only")]
MIN_GROUP, MIN_CLASSIFIER = 5, 10
N_PERM, SEED = 999, 0
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


# --------------------------------------------------------------------------
# Statistics (computation; no printing or files)
# --------------------------------------------------------------------------
def r2(X, g):
    """Share of the total sum of squares lying between the groups of g (0 = none, 1 = all).
    X (n, 204), g (n,) group labels. numpy float64, not torch: the p-values compare shuffled R2 with
    the observed one (v >= obs), so last-digit summation differences could flip a tie."""
    gm = X.mean(0)
    between = sum((g == k).sum() * ((X[g == k].mean(0) - gm) ** 2).sum() for k in np.unique(g))
    return float(between / ((X - gm) ** 2).sum())


def r2_test(X, g, rows, rng):
    """R2, permutation p (labels shuffled within rows) and the mean shuffled R2 (chance).

    Shuffling inside each row keeps every row's mix of groups, so a row effect (rows differ in
    prevalence and spectrally) stays in the null and cannot pass for disease. p = (1 + #shuffles with
    R2 >= observed) / (1 + N_PERM). numpy rng: its stream gives the reported p-values."""
    obs, null = r2(X, g), []
    for _ in range(N_PERM):
        gp = g.copy()
        for r in np.unique(rows):
            idx = np.flatnonzero(rows == r)
            gp[idx] = rng.permutation(g[idx])
        null.append(r2(X, gp))
    return obs, (1 + sum(v >= obs for v in null)) / (1 + N_PERM), float(np.mean(null))


def cohens_d(a, b):
    """Per band: (mean of b - mean of a) / pooled leaf-to-leaf SD. a (n_a, 204), b (n_b, 204) -> (204,).
    numpy: the peak values go into the results files."""
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean(0) - a.mean(0)) / sp


def model(C):
    """Per-band standardisation + logistic regression with balanced class weights and regularisation C.
    The scaler is inside the pipeline, so it is fitted on the training leaves only. sklearn, not torch:
    it is the reported model."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=10_000))


def loro_scores(X, y, rows, C):
    """Leave-one-row-out probability of group 1 for one C (whole rows held out: neighbouring vines look alike)."""
    s = np.empty(len(y))
    for r in np.unique(rows):
        te = rows == r
        s[te] = model(C).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return s


def nested_auc(X, y, rows):
    """Leave-one-row-out scores with C picked by an inner leave-one-row-out on the training rows
    (class-balanced log-loss); AUC computed within rows. Also returns the chosen C per test row.

    Nested: the test row never takes part in choosing C, otherwise the AUC would be optimistic.
    Within-row AUC only compares leaves of the same row, so row differences cannot create signal."""
    s, chosen = np.empty(len(y)), {}
    for r in np.unique(rows):
        te = rows == r
        Xtr, ytr, rtr = X[~te], y[~te], rows[~te]
        # 1. inner CV picks C; weights make both groups count equally in the log-loss
        w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
        inner = {C: log_loss(ytr, loro_scores(Xtr, ytr, rtr, C), sample_weight=w) for C in C_GRID}
        best = min(inner, key=inner.get)
        chosen[str(r)] = best
        # 2. refit with that C on all training rows, predict the held-out row
        s[te] = model(best).fit(Xtr, ytr).predict_proba(X[te])[:, 1]
    return float(_stratified_auc(y, s, rows)), chosen


def holm(p):
    """Holm-adjusted p-values (controls the chance of any false positive across all tests).

    Running many tests makes some small p-values likely by luck. Holm sorts the p-values and multiplies
    the k-th smallest (k = 1..n) by (n - k + 1), keeping the result non-decreasing and <= 1.
    numpy, not torch: it adjusts the 26 reported p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------
# One variety, then all of them (computation)
# --------------------------------------------------------------------------
def analyse_variety(v, X, Xs, meta, w, rng):
    """Every comparison inside one variety (the pairs and the 3-group test), raw and SNV.

    X, Xs (998, 204) raw and SNV spectra of all leaves, w (204,) nm, rng the shared numpy generator
    (comparisons run in a fixed order, which fixes the random numbers each test gets).
    Returns (records, plot): one dict per comparison, and what figure() needs for this variety."""
    m = (meta.variety == v).to_numpy() & meta.cls.isin(list(GROUPS)).to_numpy()
    grp = meta.cls.to_numpy()[m]
    names = np.array([GROUPS[c] for c in grp])
    rows = meta.row.to_numpy()[m]
    n = {g: int((names == g).sum()) for g in GROUPS.values()}
    present = [g for g in GROUPS.values() if n[g] >= MIN_GROUP]
    comparisons = [p for p in PAIRS if p[0] in present and p[1] in present]
    if len(present) == 3:
        comparisons.append(("healthy", "RB only", "LR only"))
    out, d_snv = [], {}
    for comp in comparisons:
        sel = np.isin(names, comp)
        g = np.array([comp.index(x) for x in names[sel]])
        rec = {"variety": v, "comparison": " vs ".join(comp), "n": " / ".join(str(n[x]) for x in comp)}
        for rep, A in (("raw", X[m][sel]), ("SNV", Xs[m][sel])):
            R2, p, chance = r2_test(A, g, rows[sel], rng)
            rec |= {f"R2 {rep}": R2, f"chance {rep}": chance, f"p {rep}": p}
            if len(comp) == 2:
                d = cohens_d(A[g == 0], A[g == 1])
                k = int(np.abs(d).argmax())
                rec |= {f"max |d| {rep}": float(abs(d[k])), f"at nm {rep}": float(w[k])}
                if rep == "SNV":
                    d_snv[rec["comparison"]] = d
        if len(comp) == 2 and min(n[comp[0]], n[comp[1]]) >= MIN_CLASSIFIER:
            rec["AUC (SNV classifier)"], rec["chosen C"] = nested_auc(Xs[m][sel], g, rows[sel])
        out.append(rec)
    plot = {"A": X[m], "As": Xs[m], "names": names, "d_snv": d_snv, "n": n}
    return out, plot


def add_holm(table):
    """Adds 'p Holm raw' / 'p Holm SNV' to every record, corrected over all raw and SNV tests together.
    Returns the number of tests."""
    ps = [t[f"p {rep}"] for t in table for rep in ("raw", "SNV")]
    adj = iter(holm(ps))
    for t in table:
        t["p Holm raw"], t["p Holm SNV"] = next(adj), next(adj)
    return len(ps)


def run_all(X, Xs, meta, w):
    """Every variety in order with one seeded rng, then the Holm correction.
    Returns (table of records, {variety: plot data}, number of tests)."""
    rng = np.random.default_rng(SEED)                              # numpy stream behind the reported p-values
    table, plots = [], {}
    for v in VARIETIES:
        recs, plots[v] = analyse_variety(v, X, Xs, meta, w, rng)
        table += recs
    return table, plots, add_holm(table)


# --------------------------------------------------------------------------
# Reporting: figures and result files
# --------------------------------------------------------------------------
def figure(v, A, As, names, d_snv, w, n):
    """figures/<variety>.png: group means (raw, SNV), per-band Cohen's d (SNV), PCA (plot only; sklearn)."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for panel, S, title, ylab in [(ax[0, 0], A, "(a) raw mean spectra (+- 1 SD)", "reflectance"),
                                  (ax[0, 1], As, "(b) SNV spectra: brightness removed (+- 1 SD)", "SNV value")]:
        for g, col in COLORS.items():
            if n[g] == 0:
                continue
            mu, sd = S[names == g].mean(0), S[names == g].std(0)
            panel.fill_between(w, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            panel.plot(w, mu, color=col, lw=2, label=f"{g} (n={n[g]})")
        panel.set(title=title, xlabel="wavelength (nm)", ylabel=ylab, xlim=(w[0], w[-1]))
        panel.legend(loc="upper left")
    styles = {"healthy vs RB only": (COLORS["RB only"], "-"), "healthy vs LR only": (COLORS["LR only"], "-"),
              "RB only vs LR only": ("k", "--")}
    for name, d in d_snv.items():
        col, ls = styles[name]
        ax[1, 0].plot(w, d, color=col, ls=ls, lw=1.8, label=f"{name} (d = 2nd - 1st)")
    for y, lab in ((0.5, "medium"), (0.8, "large")):
        for s in (-1, 1):
            ax[1, 0].axhline(s * y, color="grey", ls=":", lw=0.8)
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set(title="(c) per-band effect size, Cohen's d (SNV); dotted = |d| 0.5 and 0.8",
                 xlabel="wavelength (nm)", ylabel="Cohen's d", xlim=(w[0], w[-1]), ylim=(-1.2, 1.2))
    ax[1, 0].legend(loc="best", fontsize=8)
    Z = PCA(2).fit(As).transform(As)
    for g, col in COLORS.items():
        if n[g]:
            ax[1, 1].scatter(Z[names == g, 0], Z[names == g, 1], s=14, alpha=0.7, color=col, lw=0, label=g)
    ax[1, 1].set(title=f"(d) PCA of {v}'s SNV spectra", xlabel="PC1", ylabel="PC2")
    ax[1, 1].legend(loc="best", markerscale=1.5)
    fig.suptitle(f"{v}: healthy vs red blotch vs leafroll (leaves with both viruses left out)")
    fig.tight_layout()
    fig.savefig(FIG / f"{v.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def summary_figure(table):
    """figures/summary_effect_sizes.png: the largest |d| of every pair comparison, raw and SNV."""
    pairs = [t for t in table if " vs " in t["comparison"] and t["comparison"].count(" vs ") == 1]
    labels = [f"{t['variety']}: {t['comparison']} (n {t['n']})" for t in pairs]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.9), 6.5))
    ax.bar(x - 0.2, [t["max |d| raw"] for t in pairs], 0.4, label="raw", color="#999999")
    ax.bar(x + 0.2, [t["max |d| SNV"] for t in pairs], 0.4, label="SNV", color="#0072B2")
    for t, xi in zip(pairs, x):
        if t["p Holm SNV"] < 0.05 or t["p Holm raw"] < 0.05:
            ax.text(xi, max(t["max |d| raw"], t["max |d| SNV"]) + 0.02, "*", ha="center", fontsize=14)
    for y in (0.5, 0.8):
        ax.axhline(y, color="grey", ls=":", lw=0.8)
    ax.set_xticks(x, labels, fontsize=8, rotation=35, ha="right")
    ax.set(ylabel="largest |Cohen's d| over the 204 bands",
           title="Largest per-band disease effect inside each variety (* = Holm-adjusted p < 0.05)\n"
                 "small groups (e.g. 9 healthy Cab Franc leaves) give large d by chance - check p")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "summary_effect_sizes.png", dpi=150)
    plt.close(fig)


def disease_markdown(table, n_tests) -> str:
    """results/disease_differences.md: one table for raw and one for SNV spectra (** = Holm p < 0.05)."""
    L = ["# Healthy vs red blotch vs leafroll, inside each variety", "",
         "One mean spectrum per leaf; leaves with both viruses left out. R2 = share of the variety's spectral "
         "variance between the groups; chance = average R2 of shuffled labels; p = 999 shuffles within rows; "
         f"p Holm = corrected for all {n_tests} tests (use this one: < 0.05 = significant). max |d| = largest "
         "per-band Cohen's d (0.5 medium, 0.8 large). AUC = leave-one-row-out logistic regression on SNV "
         "spectra, within rows (0.5 = guessing; '-' = a group has < 10 leaves).", ""]
    for rep in ("raw", "SNV"):
        L += [f"## {rep} spectra", "", "| variety | comparison | n | R2 (chance) | p | p Holm | max abs d (nm) |"
              + (" AUC |" if rep == "SNV" else ""), "|---|---|---|---|---|---|---|" + ("---|" if rep == "SNV" else "")]
        for t in table:
            d = f"{t[f'max |d| {rep}']:.2f} ({t[f'at nm {rep}']:.0f})" if f"max |d| {rep}" in t else "-"
            sig = "**" if t[f"p Holm {rep}"] < 0.05 else ""
            line = (f"| {t['variety']} | {t['comparison']} | {t['n']} | {t[f'R2 {rep}']:.3f} ({t[f'chance {rep}']:.3f}) | "
                    f"{t[f'p {rep}']:.3f} | {sig}{t[f'p Holm {rep}']:.3f}{sig} | {d} |")
            if rep == "SNV":
                line += f" {t['AUC (SNV classifier)']:.3f} |" if "AUC (SNV classifier)" in t else " - |"
            L.append(line)
        L.append("")
    return "\n".join(L)


def write_outputs(table, plots, n_tests, w) -> str:
    """All figures plus results/disease_differences.csv / .json / .md. Returns the markdown for the log."""
    for v, p in plots.items():
        figure(v, p["A"], p["As"], p["names"], p["d_snv"], w, p["n"])
    summary_figure(table)
    pd.DataFrame(table).to_csv(RES / "disease_differences.csv", index=False)
    (RES / "disease_differences.json").write_text(json.dumps(table, indent=2))
    md = disease_markdown(table, n_tests)
    (RES / "disease_differences.md").write_text(md)
    return md


def main():
    """Load leaf spectra -> every variety's tests + Holm correction -> figures, result files, logs/run.log."""
    for d in (FIG, RES, LOGS):
        d.mkdir(exist_ok=True)
    (LOGS / "run.log").unlink(missing_ok=True)                     # the log holds the latest run only
    log = get_logger("disease_differences", log_file=LOGS / "run.log")

    meta = load_meta()
    meta["cls"] = meta.red_blotch + 2 * meta.leafroll3              # 0 healthy, 1 RB only, 2 LR only, 3 both
    X, _ = leaf_mean_spectra(meta)
    Xs, w = snv(X), load_wavelengths()

    table, plots, n_tests = run_all(X, Xs, meta, w)

    md = write_outputs(table, plots, n_tests, w)
    log.info(md)


if __name__ == "__main__":
    main()
