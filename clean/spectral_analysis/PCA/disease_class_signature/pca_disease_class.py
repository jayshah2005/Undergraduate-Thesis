"""Ask whether SNV leaf-mean spectra differ by disease class."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pca_common import (  # noqa: E402
    CLASS_COLORS, CLASSES, N_PCS, SEED, VAR_MARKERS, VARIETIES, between_ss,
    between_ss_within, cohens_d, get_device, get_logger, load_leaf_means, np_,
    pca_fit, r2, r2_test, snv, total_ss,
)

FIG, RES = HERE / "figures", HERE / "results"
CONTRASTS = [("RB only", 1, CLASS_COLORS[1]), ("LR only", 2, CLASS_COLORS[2]),
             ("both", 3, CLASS_COLORS[3]), ("any disease", None, "k")]


# Computation
def variance_share(X: torch.Tensor, var: torch.Tensor, cls: torch.Tensor, row: torch.Tensor) -> dict:
    """Split SNV variance into variety vs disease vs row.

    The last two overlap because disease prevalence differs by row.

    n: the number of vines.

    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    :param row: Vineyard-row codes, length n.
    :type row: torch.Tensor
    :return: R² for each grouping.
    :rtype: dict
    """
    tot = total_ss(X)
    return {
        "variety": float(between_ss(X, var) / tot), # Share of variance between varieties
        "class (pooled over varieties)": float(between_ss(X, cls) / tot), # Share of variance between disease classes when all varieties are pooled together
        "class within variety": float(between_ss_within(X, cls, var) / tot), # Share of variance between disease classes within each variety
        "row within variety": float(between_ss_within(X, row, var) / tot), # Share of variance between rows within each variety
    }


def analyse_within(Xv: torch.Tensor, cv: torch.Tensor, rv: torch.Tensor,
                   generator: torch.Generator) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Test the 4-class label (and coarser targets) plus PCA on this variety only.

    n_v: the number of vines in this variety.
    n_pcs: the number of principal components returned.

    :param Xv: SNV spectra for one variety, shape (n_v, 204).
    :type Xv: torch.Tensor
    :param cv: Disease-class codes for that variety, length n_v.
    :type cv: torch.Tensor
    :param rv: Vineyard-row codes for that variety, length n_v.
    :type rv: torch.Tensor
    :param generator: Seeded CPU generator for the permutation test.
    :type generator: torch.Generator
    :return: Result dict, scores Z of shape (n_v, n_pcs), explained-variance
        ratio of length n_pcs.
    :rtype: tuple
    """

    # Define targets to test
    targets = {
        "class (4)": cv,    # Healthy, RB only, LR only, both
        "healthy vs any disease": (cv > 0).to(torch.long), # Healthy vs any disease
        "RB label": ((cv == 1) | (cv == 3)).to(torch.long), # RB or not
        "LR label": ((cv == 2) | (cv == 3)).to(torch.long), # LR or not
    }

    # Compute R² for each target
    res = {}
    for name, g in targets.items(): # g is the class code for target name
        r, p, chance = r2_test(Xv, g, rv, generator) # Test R² for target g
        res[name] = {"R2": r, "p (permuted within rows)": p, "R2 by chance": chance}
    
    # Test R² for row
    r, p, chance = r2_test(Xv, rv, None, generator)
    res["row"] = {"R2": r, "p (free permutation)": p, "R2 by chance": chance}

    # Fit PCA and compute eta² for each PC
    Z, ev, _loadings, _ev_all = pca_fit(Xv, N_PCS)
    res["pca"] = [
        {"pc": k + 1, "explained": float(ev[k]),
         "eta2 class": r2(Z[:, k:k + 1], cv), "eta2 row": r2(Z[:, k:k + 1], rv)}
        for k in range(N_PCS)
    ]

    return res, Z, ev # Return results, scores Z of shape (n_v, n_pcs), explained-variance ratio of length n_pcs

def effect_curves(X: torch.Tensor, var: torch.Tensor, cls: torch.Tensor) -> dict:
    """Compute per-band Cohen's d of healthy vs each contrast, inside every variety.

    n: the number of vines.

    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    :return: Nested dict variety -> contrast name -> (204,) d.
    :rtype: dict
    """
    out = {}
    for i, v in enumerate(VARIETIES):
        healthy = X[(var == i) & (cls == 0)]
        curves = {}
        for name, c, _col in CONTRASTS:
            if c is None:
                diseased = X[(var == i) & (cls > 0)]
            else:
                diseased = X[(var == i) & (cls == c)]
            if healthy.shape[0] < 2 or diseased.shape[0] < 2:
                continue
            curves[name] = cohens_d(healthy, diseased)
        out[v] = curves
    return out


def effect_peaks(w, curves: dict) -> dict:
    """Record the largest |d| and its wavelength for every variety x contrast.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param curves: Output of :func:`effect_curves`; each curve is length 204.
    :type curves: dict
    :return: Comparison name -> max |d| and wavelength.
    :rtype: dict
    """
    peaks = {}
    for v, ds in curves.items():
        for name, d in ds.items():
            k = int(torch.abs(d).argmax())
            peaks[f"{v}: healthy vs {name}"] = {
                "max |d|": float(torch.abs(d)[k]), "at nm": float(w[k]),
            }
    return peaks


# Figures
def fig_spectra_by_class(w, X, var, cls):
    """Plot class mean ± 1 SD inside each variety (SNV).

    n: the number of vines.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, i, v in zip(axes.flat, range(4), VARIETIES):
        for c, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
            m = (var == i) & (cls == c)
            if not m.any():
                continue
            mu, sd = np_(X[m].mean(0)), np_(X[m].std(0, unbiased=False))
            ax.fill_between(w, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.plot(w, mu, color=col, lw=2, label=f"{name} (n={int(m.sum())})")
        ax.set(title=v, xlim=(w[0], w[-1]))
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SNV value")
    fig.suptitle("Disease-class mean SNV spectra within each variety (shading = ± 1 SD over vines)")
    fig.tight_layout()
    fig.savefig(FIG / "spectra_by_class_per_variety.png", dpi=150)
    plt.close(fig)


def fig_difference_healthy(w, X, var, cls):
    """Plot class mean minus healthy mean for Baco and Chardonnay.

    n: the number of vines.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, i, v in zip(axes, (0, 2), ("Baco", "Chardonnay")):
        healthy = X[(var == i) & (cls == 0)].mean(0)
        for c in (1, 2, 3):
            m = (var == i) & (cls == c)
            if not m.any():
                continue
            ax.plot(w, np_(X[m].mean(0) - healthy), color=CLASS_COLORS[c], lw=2,
                    label=f"{CLASSES[c]} (n={int(m.sum())})")
        ax.axhline(0, color="k", lw=0.6)
        n_h = int(((var == i) & (cls == 0)).sum())
        ax.set(title=f"{v}: class mean − healthy (n={n_h})", xlabel="wavelength (nm)",
               xlim=(w[0], w[-1]))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("SNV difference")
    fig.suptitle("Disease difference inside a variety, SNV (same y-scale on both panels)")
    fig.tight_layout()
    fig.savefig(FIG / "difference_from_healthy.png", dpi=150)
    plt.close(fig)


def fig_pca_pooled(Z, ev, var, cls):
    """Plot all leaves with colour = disease class. Confounded: class tracks cultivar.

    n: the number of vines.
    n_pcs: the number of principal components returned.

    :param Z: Scores from a pooled PCA, shape (n, n_pcs).
    :type Z: torch.Tensor
    :param ev: Explained-variance ratio of those PCs, length n_pcs.
    :type ev: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    """
    z, e = np_(Z), np_(ev)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (i, j) in zip(axes, [(0, 1), (1, 2)]):
        for c, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
            for k, v in enumerate(VARIETIES):
                m = (np_(cls) == c) & (np_(var) == k)
                ax.scatter(z[m, i], z[m, j], s=12, alpha=0.6, color=col,
                           marker=VAR_MARKERS[v], lw=0,
                           label=name if k == 0 else None)
        ax.set(xlabel=f"PC{i + 1} ({e[i]:.1%})", ylabel=f"PC{j + 1} ({e[j]:.1%})")
        ax.legend(loc="best", markerscale=1.8, fontsize=8)
    fig.text(0.5, 0.01,
             "Pooled class colouring partly repeats variety: Cab Franc is 96% RB only, Vidal 63%.",
             ha="center", fontsize=9, style="italic")
    fig.suptitle("PCA of SNV leaf-mean spectra (all 998 vines), coloured by disease class")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "pca_pooled_by_class.png", dpi=150)
    plt.close(fig)


def fig_pca_within(X, var, cls):
    """Fit a new PCA inside each variety so cultivar cannot count as disease.

    n: the number of vines.

    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param cls: Disease-class codes, length n.
    :type cls: torch.Tensor
    """
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for r, v in enumerate(VARIETIES):
        m = var == r
        Z, ev, _loadings, _ev_all = pca_fit(X[m], N_PCS)
        z, e, cv = np_(Z), np_(ev), np_(cls[m])
        for col, (i, j) in enumerate([(0, 1), (1, 2)]):
            ax = axes[r, col]
            for c, (name, colr) in enumerate(zip(CLASSES, CLASS_COLORS)):
                ax.scatter(z[cv == c, i], z[cv == c, j], s=14, alpha=0.7, color=colr, lw=0,
                           label=f"{name} (n={int((cv == c).sum())})")
            ax.set(title=f"{v} only: PC{i + 1} vs PC{j + 1}",
                   xlabel=f"PC{i + 1} ({e[i]:.1%})", ylabel=f"PC{j + 1} ({e[j]:.1%})")
            ax.legend(loc="best", markerscale=1.4, fontsize=7)
    fig.suptitle("PCA fitted inside one variety (SNV), coloured by disease class")
    fig.tight_layout()
    fig.savefig(FIG / "pca_within_variety.png", dpi=150)
    plt.close(fig)


def fig_effect_sizes(w, curves: dict):
    """Plot Cohen's d vs wavelength, one panel per variety.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param curves: Output of :func:`effect_curves`; each curve is length 204.
    :type curves: dict
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, v in zip(axes.flat, VARIETIES):
        for name, _c, colr in CONTRASTS:
            if name not in curves[v]:
                continue
            ax.plot(w, np_(curves[v][name]), color=colr, lw=2, label=f"healthy vs {name}")
        for y in (-0.5, 0.5):
            ax.axhline(y, color="grey", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(title=v, xlabel="wavelength (nm)", xlim=(w[0], w[-1]))
        ax.legend(loc="best", fontsize=8)
    axes[0, 0].set_ylabel("Cohen's d (diseased − healthy)")
    axes[1, 0].set_ylabel("Cohen's d (diseased − healthy)")
    fig.suptitle("Per-band SNV effect size of disease within variety (dotted = |d| 0.5)")
    fig.tight_layout()
    fig.savefig(FIG / "effect_size.png", dpi=150)
    plt.close(fig)


# Report
def f3(x: float) -> str:
    """Format a float to three decimal places.

    :param x: Value to format.
    :type x: float
    :return: The number as a string with three decimals.
    :rtype: str
    """
    return f"{x:.3f}"


def tables(res: dict) -> str:
    """Build markdown of variance shares, within-variety tests, and peak effect sizes.

    :param res: Output of :func:`compute`.
    :type res: dict
    :return: Markdown text.
    :rtype: str
    """
    sn = res["snv"]
    lines = [
        "# Is the spectral signature different by disease class?",
        "",
        "SNV-normalised leaf-mean spectra (brightness removed). Classes: healthy / RB only / "
        "LR only / both. R² is the share of spectral variance between those groups. Disease "
        "p-values permute labels *within vineyard rows*. Pooled class R² is inflated by variety "
        "(Cab Franc ≈ 96% RB only).",
        "",
        "## Share of total variance (SNV)",
        "",
        "| grouping | R² |",
        "|---|---|",
    ]
    for k, v in sn["variance_share"].items():
        lines.append(f"| {k} | {f3(v)} |")
    lines.append("")
    for v in VARIETIES:
        a = sn["within"][v]
        lines += [
            f"## {v}: does disease separate the vines?",
            "",
            "| target | R² | chance | p |",
            "|---|---|---|---|",
        ]
        for t in ("class (4)", "healthy vs any disease", "RB label", "LR label"):
            lines.append(
                f"| {t} | {f3(a[t]['R2'])} | {f3(a[t]['R2 by chance'])} | "
                f"{a[t]['p (permuted within rows)']:.3f} |"
            )
        lines.append(
            f"| row (for scale; free permutation) | {f3(a['row']['R2'])} | "
            f"{f3(a['row']['R2 by chance'])} | {a['row']['p (free permutation)']:.3f} |"
        )
        lines.append("")
        lines.append(
            "PCA eta² of class / row per PC: "
            + ", ".join(
                f"PC{p['pc']} {p['eta2 class']:.2f}/{p['eta2 row']:.2f}" for p in a["pca"]
            )
            + "."
        )
        lines.append("")
    lines += [
        "## Largest per-band effect sizes (Cohen's d, SNV)",
        "",
        "| comparison | max abs d | at nm |",
        "|---|---|---|",
    ]
    for k, v in res["effect_size_peaks"].items():
        lines.append(f"| {k} | {v['max |d|']:.2f} | {v['at nm']:.0f} |")
    return "\n".join(lines) + "\n"


def compute(device: torch.device):
    """SNV-normalise spectra and run the disease-class analysis with a seeded generator.

    n: the number of vines.
    n_pcs: the number of principal components returned.

    :param device: GPU or CPU.
    :type device: torch.device
    :return: Result dict, wavelengths (204,), SNV spectra (n, 204), variety and
        class codes of length n each, pooled scores Z of shape (n, n_pcs),
        explained-variance ratio of length n_pcs, effect curves (each d of
        length 204).
    :rtype: tuple
    """
    # Set seeds to ensure reproducibility
    torch.manual_seed(SEED) # Set global seed
    generator = torch.Generator() # Create generator
    generator.manual_seed(SEED) # Set generator seed

    # Load data
    generator = torch.Generator()
    generator.manual_seed(SEED)
    _meta, w, X_raw, _glint, var, cls, row = load_leaf_means(device)

    # SNV-normalise spectra
    X = snv(X_raw)

    share = variance_share(X, var, cls, row)
    within = {}
    for i, v in enumerate(VARIETIES):
        m = var == i
        stats, _Z, _ev = analyse_within(X[m], cls[m], row[m], generator) # Test R² for each target and do PCA inside each variety
        within[v] = stats

    Z_all, ev_all, _loadings, _ev = pca_fit(X, N_PCS) # Fit PCA to all vines

    curves = effect_curves(X, var, cls)
    res = {
        "n_leaves": int(X.shape[0]),
        "device": str(device),
        "snv": {"variance_share": share, "within": within},
        "effect_size_peaks": effect_peaks(w, curves),
    }
    return res, w, X, var, cls, Z_all, ev_all, curves


def main() -> None:
    """Run the analysis and write figures, tables, and the log."""
    # Create directories
    for d in (FIG, RES):
        d.mkdir(exist_ok=True)
    log = get_logger("pca_disease_class")

    # Get device and compute
    device = get_device()

    # Compute
    res, w, X, var, cls, Z, ev, curves = compute(device)

    # Write results
    md = tables(res)
    (RES / "tables.md").write_text(md)
    fig_spectra_by_class(w, X, var, cls)
    fig_difference_healthy(w, X, var, cls)
    fig_pca_pooled(Z, ev, var, cls)
    fig_pca_within(X, var, cls)
    fig_effect_sizes(w, curves)
    log.info("[saved] results/tables.md, figures/")


if __name__ == "__main__":
    main()
