"""Which wavelengths drive the 204-band variety logistic regression?

Run from claude runs/variety_logistic_regression/:
    ../../.venv/bin/python -u band_importance.py

The headline 0.94 score uses all 204 SNV bands. Coefficients alone are a poor guide
(neighbouring bands are highly correlated, so weight spreads along the curve). This script
asks three complementary questions, all on the same 5 row folds as the main model:

  1. Univariate: logistic regression on *one* band. If one wavelength were enough, it
     would show up here.
  2. Regions: the same nested-C model on only blue / green / red / red-edge / NIR.
  3. Permutation: fit the full 204-band model, shuffle one band on the test row, measure
     the drop in balanced accuracy. A band that is redundant with its neighbours barely
     drops the score even if its coefficient is large.

Outputs
  results/band_importance.md / .json
  figures/band_importance.png
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from sklearn.feature_selection import f_classif                      # noqa: E402
from sklearn.metrics import balanced_accuracy_score                  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from common import load_wavelengths                                  # noqa: E402
from logs import get_logger                                          # noqa: E402
from variety_logistic_regression import (                            # noqa: E402
    VARIETIES, evaluate, load_features, make_model, nested_row_cv,
)

RES, FIG, LOGS = HERE / "results", HERE / "figures", HERE / "logs"
# Named slices of the 400–1000 nm curve (inclusive). RGB is the 3-band control.
REGIONS = {
    "blue 400-500 nm": (400, 500),
    "green 500-600 nm": (500, 600),
    "red 600-680 nm": (600, 680),
    "red edge 680-760 nm": (680, 760),
    "NIR 760-1000 nm": (760, 1000),
    "visible 400-700 nm": (400, 700),
}
TOP_K = (1, 3, 5, 10, 20, 50)


def band_mask(wavelengths: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (wavelengths >= lo) & (wavelengths <= hi)


def row_cv_predict(X: np.ndarray, y: np.ndarray, rows: np.ndarray, C: float) -> np.ndarray:
    """Outer row hold-out only (no inner C search). Used for the 204 one-band models."""
    pred = np.empty(len(y), dtype=int)
    for r in np.unique(rows):
        te = rows == r
        pred[te] = make_model(C).fit(X[~te], y[~te]).predict(X[te])
    return pred


def univariate_bacc(X: np.ndarray, y: np.ndarray, rows: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Balanced accuracy of a one-band logistic regression, one score per column of X."""
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        pred = row_cv_predict(X[:, [j]], y, rows, C)
        out[j] = balanced_accuracy_score(y, pred)
    return out


def permutation_drop(X: np.ndarray, y: np.ndarray, rows: np.ndarray, chosen_C: dict,
                     seed: int = 0) -> np.ndarray:
    """Mean drop in test-row balanced accuracy after shuffling one band (others left intact)."""
    rng = np.random.default_rng(seed)
    drops = np.zeros(X.shape[1])
    for r in np.unique(rows):
        te = rows == r
        clf = make_model(chosen_C[r]).fit(X[~te], y[~te])
        base = balanced_accuracy_score(y[te], clf.predict(X[te]))
        Xt = X[te]
        for j in range(X.shape[1]):
            Xp = Xt.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j] += base - balanced_accuracy_score(y[te], clf.predict(Xp))
    return drops / len(np.unique(rows))


def region_scores(X: np.ndarray, y: np.ndarray, rows: np.ndarray, wavelengths: np.ndarray) -> dict:
    """Nested row-CV balanced accuracy using only the bands in each named region."""
    out = {}
    for name, (lo, hi) in REGIONS.items():
        m = band_mask(wavelengths, lo, hi)
        P, folds = nested_row_cv(X[:, m], y, rows)
        metrics = evaluate(y, P)
        out[name] = {
            "n_bands": int(m.sum()),
            "lo_nm": lo,
            "hi_nm": hi,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "folds": {r: f["test_balanced_accuracy"] for r, f in folds.items()},
        }
    return out


def topk_scores(X: np.ndarray, y: np.ndarray, rows: np.ndarray, wavelengths: np.ndarray,
                chosen_C: dict) -> dict:
    """Per fold, pick the k most variety-separating bands on the *training* rows (ANOVA F),
    then fit logistic regression on those k. Avoids peeking at the test row."""
    out = {k: np.empty(len(y), dtype=int) for k in TOP_K}
    picked = {k: [] for k in TOP_K}
    for r in np.unique(rows):
        te = rows == r
        F, _ = f_classif(X[~te], y[~te])
        order = np.argsort(F)[::-1]
        for k in TOP_K:
            idx = np.sort(order[:k])
            picked[k].append(wavelengths[idx].tolist())
            out[k][te] = make_model(chosen_C[r]).fit(X[~te][:, idx], y[~te]).predict(X[te][:, idx])
    return {
        str(k): {
            "balanced_accuracy": balanced_accuracy_score(y, out[k]),
            "bands_nm_per_fold": picked[k],
        }
        for k in TOP_K
    }


def plot_importance(wavelengths, uni, mean_abs_coef, perm, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    series = [
        (uni, "One-band model: balanced accuracy (chance 0.25)", (0.2, 1.0)),
        (mean_abs_coef, "Full model: mean |coefficient| over varieties", None),
        (perm, "Full model: accuracy drop if that band is shuffled", None),
    ]
    for ax, (y, title, ylim) in zip(axes, series):
        ax.plot(wavelengths, y, lw=1.4, color="#0072B2")
        ax.set_ylabel("")
        ax.set_title(title, loc="left", fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
            ax.axhline(0.25, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        for x0, x1, c in ((400, 500, "#d9edf7"), (500, 600, "#dff0d8"),
                          (600, 680, "#f2dede"), (680, 760, "#fcf8e3"), (760, 1000, "#f5f5f5")):
            ax.axvspan(x0, x1, color=c, zorder=0)
    axes[-1].set_xlabel("wavelength (nm)")
    axes[-1].set_xlim(wavelengths[0], wavelengths[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(wavelengths, uni, mean_abs_coef, perm, regions, topk, full_bacc) -> str:
    top_uni = np.argsort(uni)[::-1][:8]
    top_coef = np.argsort(mean_abs_coef)[::-1][:8]
    top_perm = np.argsort(perm)[::-1][:8]

    def rows(idx):
        return "\n".join(
            f"| {wavelengths[j]:.1f} | {uni[j]:.3f} | {mean_abs_coef[j]:.3f} | {perm[j]:.4f} |"
            for j in idx
        )

    region_lines = "\n".join(
        f"| {name} | {d['n_bands']} | {d['balanced_accuracy']:.3f} | {d['macro_f1']:.3f} |"
        for name, d in regions.items()
    )
    topk_lines = "\n".join(
        f"| {k} | {d['balanced_accuracy']:.3f} |" for k, d in topk.items()
    )
    return "\n".join([
        "# Which bands drive variety prediction (SNV, 204-band model)",
        "",
        f"Full 204-band nested-CV balanced accuracy is **{full_bacc:.3f}**. "
        "Univariate scores use the same 5 row folds with C = 1. Region scores retune C inside "
        "each training split, like the main model. Permutation uses each fold's chosen C.",
        "",
        "## Regions (not a single band)",
        "",
        "| region | n bands | balanced accuracy | macro-F1 |",
        "|---|---|---|---|",
        f"| all 204 bands | 204 | {full_bacc:.3f} | — |",
        region_lines,
        "",
        "## Fewest bands (ANOVA-F selection *inside* each training split)",
        "",
        "| k best bands | balanced accuracy |",
        "|---|---|",
        topk_lines,
        "",
        "## Strongest individual wavelengths",
        "",
        "### Best one-band models",
        "",
        "| nm | univariate bacc | mean |coef| | permutation drop |",
        "|---|---|---|---|",
        rows(top_uni),
        "",
        "### Largest |coefficient| in the full model",
        "",
        "| nm | univariate bacc | mean |coef| | permutation drop |",
        "|---|---|---|---|",
        rows(top_coef),
        "",
        "### Largest permutation drop (band is not redundant with the others)",
        "",
        "| nm | univariate bacc | mean |coef| | permutation drop |",
        "|---|---|---|---|",
        rows(top_perm),
        "",
        "Shaded regions in `figures/band_importance.png`: blue, green, red, red edge, NIR.",
        "",
    ])


def main():
    for d in (RES, FIG, LOGS):
        d.mkdir(exist_ok=True)
    log_file = LOGS / "band_importance.log"
    log_file.unlink(missing_ok=True)
    log = get_logger("band_importance", log_file=log_file)

    X, y, rows, _ = load_features("snv")
    w = load_wavelengths()
    main_metrics = json.loads((RES / "variety_metrics.json").read_text())
    chosen = {r: rec["chosen_C"] for r, rec in main_metrics["folds"].items()}
    full_bacc = main_metrics["balanced_accuracy"]

    uni = univariate_bacc(X, y, rows)
    perm = permutation_drop(X, y, rows, chosen)
    c_common = Counter(chosen.values()).most_common(1)[0][0]
    coef = make_model(c_common).fit(X, y)[-1].coef_
    mean_abs_coef = np.abs(coef).mean(0)
    regions = region_scores(X, y, rows, w)
    topk = topk_scores(X, y, rows, w, chosen)

    plot_importance(w, uni, mean_abs_coef, perm, FIG / "band_importance.png")
    payload = {
        "full_balanced_accuracy": full_bacc,
        "wavelengths_nm": w.tolist(),
        "univariate_balanced_accuracy": uni.tolist(),
        "mean_abs_coefficient": mean_abs_coef.tolist(),
        "permutation_drop": perm.tolist(),
        "regions": regions,
        "top_k": topk,
        "best_univariate_nm": float(w[int(np.argmax(uni))]),
        "best_univariate_bacc": float(uni.max()),
    }
    (RES / "band_importance.json").write_text(json.dumps(payload, indent=2, default=float))
    md = markdown(w, uni, mean_abs_coef, perm, regions, topk, full_bacc)
    (RES / "band_importance.md").write_text(md)
    log.info(md)


if __name__ == "__main__":
    main()
