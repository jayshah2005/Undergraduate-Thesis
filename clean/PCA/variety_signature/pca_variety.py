"""Ask whether SNV leaf-mean spectra differ by grape variety."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pca_common import (  # noqa: E402
    N_PCS, SEED, VAR_COLORS, VAR_MARKERS, VARIETIES, get_device, get_logger,
    load_leaf_means, np_, pca_fit, pca_table, r2_test, snv,
)

FIG, RES, LOGS = HERE / "figures", HERE / "results", HERE / "logs"


# Computation
def analyse(X: torch.Tensor, var: torch.Tensor, mean_reflectance: torch.Tensor,
            generator: torch.Generator) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Compute the variance share of variety, PCA scores, and eta² on each PC.

    n: the number of vines.
    n_pcs: the number of principal components returned.

    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    :param mean_reflectance: Raw mean reflectance per vine, length n. Used to check SNV.
    :type mean_reflectance: torch.Tensor
    :param generator: Seeded CPU generator for the permutation test.
    :type generator: torch.Generator
    :return: Result dict, scores Z of shape (n, n_pcs), explained-variance
        ratio of length n_pcs.
    :rtype: tuple
    """
    Z, ev, _loadings, _ev_all = pca_fit(X, N_PCS)
    r, p, chance = r2_test(X, var, None, generator)
    out = {
        "r2_variety": r,
        "p (free permutation)": p,
        "r2_by_chance": chance,
        "pca": pca_table(Z, ev, var, mean_reflectance, "variety"),
    }
    return out, Z, ev


# Figures
def fig_spectra(w, X, var):
    """Plot every leaf's SNV curve, faint, plus the four variety means.

    n: the number of vines.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, v in enumerate(VARIETIES):
        m = var == i
        ax.plot(w, np_(X[m]).T, color=VAR_COLORS[v], lw=0.4, alpha=0.12)
    for i, v in enumerate(VARIETIES):
        m = var == i
        ax.plot(w, np_(X[m].mean(0)), color=VAR_COLORS[v], lw=2.5,
                label=f"{v} (n={int(m.sum())})",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground="white")])
    ax.set(title="SNV-normalised mean spectra", xlabel="wavelength (nm)",
           ylabel="SNV value", xlim=(w[0], w[-1]))
    ax.legend(loc="upper left")
    fig.suptitle("One mean spectrum per vine (thin) and variety means (bold)")
    fig.tight_layout()
    fig.savefig(FIG / "spectra_by_variety.png", dpi=150)
    plt.close(fig)


def fig_difference(w, X, var):
    """Plot variety mean minus the grand mean.

    n: the number of vines.

    :param w: Wavelengths in nm, shape (204,).
    :type w: torch.Tensor
    :param X: SNV spectra, shape (n, 204).
    :type X: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    gm = X.mean(0)
    for i, v in enumerate(VARIETIES):
        ax.plot(w, np_(X[var == i].mean(0) - gm), color=VAR_COLORS[v], lw=2, label=v)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title="Variety mean − grand mean (SNV)", xlabel="wavelength (nm)",
           ylabel="SNV difference", xlim=(w[0], w[-1]))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "difference_from_grand_mean.png", dpi=150)
    plt.close(fig)


def fig_pca(Z, ev, var):
    """Plot PC1 vs PC2 and PC2 vs PC3, coloured by variety.

    n: the number of vines.
    n_pcs: the number of principal components returned.

    :param Z: Scores, shape (n, n_pcs).
    :type Z: torch.Tensor
    :param ev: Explained-variance ratio of those PCs, length n_pcs.
    :type ev: torch.Tensor
    :param var: Variety codes, length n.
    :type var: torch.Tensor
    """
    z, e = np_(Z), np_(ev)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (i, j) in zip(axes, [(0, 1), (1, 2)]):
        for k, v in enumerate(VARIETIES):
            m = np_(var) == k
            ax.scatter(z[m, i], z[m, j], s=12, alpha=0.65, color=VAR_COLORS[v],
                       marker=VAR_MARKERS[v], label=v, lw=0)
        ax.set(xlabel=f"PC{i + 1} ({e[i]:.1%})", ylabel=f"PC{j + 1} ({e[j]:.1%})")
        ax.legend(loc="best", markerscale=1.8, fontsize=8)
    fig.suptitle("PCA of SNV-normalised leaf-mean spectra, coloured by variety")
    fig.tight_layout()
    fig.savefig(FIG / "pca.png", dpi=150)
    plt.close(fig)


# Report
def tables(res: dict) -> str:
    """Build markdown of the variety R² and the first 10 PCs.

    :param res: Output of :func:`compute`.
    :type res: dict
    :return: Markdown text.
    :rtype: str
    """
    r = res["snv"]
    lines = [
        "# Is the spectral signature different by variety?",
        "",
        "SNV-normalised leaf-mean spectra (mean reflectance removed). R² = share of the total "
        "sum of squares (all 204 bands) that lies between the four variety means. Chance = "
        "mean R² of shuffled labels. PCA is covariance PCA of the 998 vines.",
        "",
        f"R² variety = {r['r2_variety']:.3f} (chance {r['r2_by_chance']:.3f}, "
        f"p = {r['p (free permutation)']:.3f}).",
        "",
        f"## PCA: first {N_PCS} PCs",
        "",
        "| PC | explained | eta² variety | corr with mean_reflectance |",
        "|---|---|---|---|",
    ]
    for p in r["pca"]:
        lines.append(
            f"| {p['pc']} | {p['explained']:.1%} | {p['eta2 variety']:.3f} | "
            f"{p['corr with mean_reflectance']:+.2f} |"
        )
    lines.extend([
        "",
        "Explained — of all SNV variation across 204 bands, how much lives on that PC.",
        "",
        "η² variety — of the variation in that PC's scores only, how much sits between "
        "the four varieties.",
        "",
        "corr with mean_reflectance — Pearson r of that PC's scores with each vine's "
        "raw mean reflectance (the check that SNV worked).",
    ])
    return "\n".join(lines) + "\n"


def compute(device: torch.device):
    """SNV-normalise spectra and run the variety analysis with a seeded generator.

    n: the number of vines.
    n_pcs: the number of principal components returned.

    :param device: GPU or CPU.
    :type device: torch.device
    :return: Result dict, wavelengths (204,), SNV spectra (n, 204), variety
        codes of length n, scores Z of shape (n, n_pcs), explained-variance
        ratio of length n_pcs.
    :rtype: tuple
    """
    # Set seeds to ensure reproducibility
    torch.manual_seed(SEED) # Set global seed
    generator = torch.Generator() # Create generator
    generator.manual_seed(SEED) # Set generator seed

    _meta, wavelengths, X_raw, _glint, var, _cls, _row = load_leaf_means(device)

    # Raw mean over the 204 bands; SNV subtracts this, then we correlate PCs with it as a check.
    mean_reflectance = X_raw.mean(1)

    # SNV-normalise spectra
    X = snv(X_raw)

    # Compute 
    sn, Z, ev = analyse(X, var, mean_reflectance, generator)
    
    res = {"n_leaves": int(X.shape[0]), "device": str(device), "snv": sn}
    return res, wavelengths, X, var, Z, ev


def main() -> None:
    """Run the analysis and write figures, tables, and the log."""

    # Create directories
    for d in (FIG, RES, LOGS):
        d.mkdir(exist_ok=True)
    (LOGS / "run.log").unlink(missing_ok=True)
    log = get_logger("pca_variety", log_file=LOGS / "run.log")

    # Get device and compute
    device = get_device()
    log.info("device %s", device)

    # Compute
    res, w, X, var, Z, ev = compute(device)

    # Write results
    md = tables(res)
    (RES / "results.json").write_text(json.dumps(res, indent=2))
    (RES / "tables.md").write_text(md)
    fig_spectra(w, X, var)
    fig_difference(w, X, var)
    fig_pca(Z, ev, var)
    log.info(md)
    log.info("[saved] %s/*.png, %s, %s", FIG, RES / "tables.md", RES / "results.json")

if __name__ == "__main__":
    main()
