"""Ask which wavelengths drive the 204-band SNV variety model."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from logistic_regression import (FIG, INK, MUTED, RES, SEED, Config, evaluate, get_device,  # noqa: E402
                                 get_logger, load_features, make_model, nested_row_cv)

# Named slices of the 397-1004 nm curve, inclusive; the five colour regions cover all 204 bands.
REGIONS = {
    "blue 397-500 nm": (397, 500),
    "green 500-600 nm": (500, 600),
    "red 600-680 nm": (600, 680),
    "red edge 680-760 nm": (680, 760),
    "NIR 760-1004 nm": (760, 1004),
    "visible 397-700 nm": (397, 700),
}
TOP_K = (1, 3, 5, 10, 20, 50)
LINE = "#2a78d6"


def univariate_bacc(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Score a one-band model for every band, holding out one row at a time, with C = 1.

    n: the number of leaves. d: the number of bands.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :return: Balanced accuracy per band, length d.
    :rtype: np.ndarray
    """
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        pred = np.empty(len(y), dtype=int)
        for r in np.unique(rows):
            te = rows == r
            pred[te] = make_model(1.0).fit(X[~te][:, [j]], y[~te]).predict(X[te][:, [j]])
        out[j] = balanced_accuracy_score(y, pred)
    return out


def permutation_drop(X: np.ndarray, y: np.ndarray, rows: np.ndarray, chosen_C: dict) -> np.ndarray:
    """Measure how far test-row accuracy falls when one band is shuffled among the test leaves.

    A band that is redundant with its neighbours barely moves the score, even with a
    large coefficient.

    n: the number of leaves. d: the number of bands.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param chosen_C: The C of each held-out row.
    :type chosen_C: dict
    :return: Mean drop in balanced accuracy per band, length d.
    :rtype: np.ndarray
    """
    g = torch.Generator().manual_seed(SEED)
    drops = np.zeros(X.shape[1])
    for r in np.unique(rows):
        te = rows == r
        clf = make_model(chosen_C[str(r)]).fit(X[~te], y[~te])
        base = balanced_accuracy_score(y[te], clf.predict(X[te]))
        for j in range(X.shape[1]):
            shuffled = X[te].copy()
            shuffled[:, j] = shuffled[torch.randperm(len(shuffled), generator=g).numpy(), j]
            drops[j] += base - balanced_accuracy_score(y[te], clf.predict(shuffled))
    return drops / len(np.unique(rows))


def region_scores(X: np.ndarray, y: np.ndarray, rows: np.ndarray, w: np.ndarray, device: torch.device) -> dict:
    """Run the full nested model on only the bands of each named region.

    n: the number of leaves. d: the number of bands.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param w: Band centres in nm, length d.
    :type w: np.ndarray
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: {region: n_bands, balanced accuracy}.
    :rtype: dict
    """
    out = {}
    for name, (lo, hi) in REGIONS.items():
        keep = (w >= lo) & (w <= hi)
        P, _folds = nested_row_cv(X[:, keep], y, rows)
        m = evaluate(y, P, device)
        out[name] = {"n_bands": int(keep.sum()), "balanced_accuracy": m["balanced_accuracy"]}
    return out


def topk_scores(X: np.ndarray, y: np.ndarray, rows: np.ndarray, chosen_C: dict) -> dict:
    """Pick the k most variety-separating bands on the training rows, then fit on those k.

    Why scikit-learn: f_classif is the standard ANOVA F, and the score it feeds is a sklearn model.

    n: the number of leaves. d: the number of bands.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param chosen_C: The C of each held-out row.
    :type chosen_C: dict
    :return: {k: balanced accuracy}.
    :rtype: dict
    """
    pred = {k: np.empty(len(y), dtype=int) for k in TOP_K}
    for r in np.unique(rows):
        te = rows == r
        order = np.argsort(f_classif(X[~te], y[~te])[0])[::-1]
        for k in TOP_K:
            idx = np.sort(order[:k])
            pred[k][te] = make_model(chosen_C[str(r)]).fit(X[~te][:, idx], y[~te]).predict(X[te][:, idx])
    return {k: balanced_accuracy_score(y, pred[k]) for k in TOP_K}


def analyse(d: dict, device: torch.device) -> dict:
    """Run the four band questions on the all-bands features.

    :param d: Output of :func:`logistic_regression.load_features`.
    :type d: dict
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: Scores per band (each length 204), per region and per k, and the full model's score.
    :rtype: dict
    """
    X, y, rows, w = d["X"], d["y"], d["rows"], d["wavelengths"]
    P, folds = nested_row_cv(X, y, rows)
    chosen = {r: f["chosen_C"] for r, f in folds.items()}
    common_C = Counter(chosen.values()).most_common(1)[0][0]
    coef = make_model(common_C).fit(X, y)[-1].coef_
    return {
        "full_balanced_accuracy": evaluate(y, P, device)["balanced_accuracy"],
        "univariate": univariate_bacc(X, y, rows),
        "mean_abs_coef": np.abs(coef).mean(0),
        "permutation_drop": permutation_drop(X, y, rows, chosen),
        "regions": region_scores(X, y, rows, w, device),
        "top_k": topk_scores(X, y, rows, chosen),
    }


def importance_markdown(r: dict, w: np.ndarray) -> str:
    """Build band_importance.md.

    :param r: Output of :func:`analyse`.
    :type r: dict
    :param w: Band centres in nm, length 204.
    :type w: np.ndarray
    :return: Markdown text.
    :rtype: str
    """
    uni, perm = r["univariate"], r["permutation_drop"]

    def table(score: np.ndarray) -> list[str]:
        """Rows for the 8 bands with the largest score."""
        return ["| nm | one-band accuracy | permutation drop |", "|---|---|---|"] + [
            f"| {w[j]:.1f} | {uni[j]:.3f} | {perm[j]:.4f} |" for j in np.argsort(score)[::-1][:8]]

    lines = ["# Which bands drive variety prediction (all bands)", "",
             "Balanced accuracy; chance 0.25.", "",
             "## Regions", "", "| region | bands | balanced accuracy |", "|---|---|---|",
             f"| all | 204 | {r['full_balanced_accuracy']:.3f} |"]
    lines += [f"| {n} | {v['n_bands']} | {v['balanced_accuracy']:.3f} |" for n, v in r["regions"].items()]
    lines += ["", "## Fewest bands (picked by ANOVA F inside each training split)", "",
              "| k best bands | balanced accuracy |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in r["top_k"].items()]
    lines += ["", "## Best one-band models", "", *table(uni), "", "## Largest permutation drop", "", *table(perm), ""]
    return "\n".join(lines)


def plot_importance(r: dict, w: np.ndarray, path: Path) -> None:
    """Draw the three per-band scores on a shared wavelength axis, one panel each.

    :param r: Output of :func:`analyse`.
    :type r: dict
    :param w: Band centres in nm, length 204.
    :type w: np.ndarray
    :param path: Output file.
    :type path: Path
    """
    panels = [(r["univariate"], "One-band model: balanced accuracy (dashed = chance, 0.25)"),
              (r["mean_abs_coef"], "Full model: mean absolute coefficient over varieties"),
              (r["permutation_drop"], "Full model: accuracy drop when that band is shuffled")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (y, title) in zip(axes, panels):
        ax.plot(w, y, lw=2, color=LINE)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        # Light boundaries of the colour regions, labelled on the top panel only.
        for x in (500, 600, 680, 760):
            ax.axvline(x, color="#e1e0d9", lw=1, zorder=0)
    axes[0].axhline(0.25, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylim(0.2, max(0.8, float(r["univariate"].max()) + 0.05))
    for x, name in zip((448, 550, 640, 720, 880), ("blue", "green", "red", "red edge", "NIR")):
        axes[0].text(x, axes[0].get_ylim()[1], name, ha="center", va="top", color=MUTED, fontsize=9)
    axes[-1].set_xlabel("wavelength (nm)", color=MUTED)
    axes[-1].set_xlim(w[0], w[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load the all-bands features → answer the band questions → write the table and figure."""
    log = get_logger("band_importance")
    device = get_device()
    d = load_features(Config("all"), device)
    r = analyse(d, device)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    (RES / "band_importance.md").write_text(importance_markdown(r, d["wavelengths"]))
    plot_importance(r, d["wavelengths"], FIG / "band_importance.png")
    log.info("[saved] results/band_importance.md, figures/band_importance.png")


if __name__ == "__main__":
    main()
