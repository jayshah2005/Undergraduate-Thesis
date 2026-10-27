"""Predict the variety of every leaf from three pseudo-RGB bands, same model as the 204-band run.

Run from claude runs/variety_logistic_regression_rgb/:
    ../../.venv/bin/python -u variety_logistic_regression_rgb.py
    ../../.venv/bin/python -u variety_logistic_regression_rgb.py --spectra raw

This is the variety_logistic_regression pipeline with the 204-band mean spectrum replaced by
the three Specim IQ bands nearest 640 / 550 / 460 nm (R, G, B) — the same picks used as the
pseudo-RGB adaptor in the archived ResNet runs. Nested row-fold CV, SNV, C search and metrics
are unchanged, so the scores are comparable to the full-spectrum run.

Outputs (results/, suffix _raw when --spectra raw) match the 204-band folder. The coefficient
figure is a bar chart over the three bands rather than a 204-point line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE.parent / "variety_logistic_regression"))
from common import load_meta, load_wavelengths                       # noqa: E402
from logs import get_logger                                          # noqa: E402
from spectra import leaf_mean_spectra, snv                           # noqa: E402
from variety_logistic_regression import (                            # noqa: E402
    VARIETIES, evaluate, fit_coefficient_model, nested_row_cv,
    plot_confusion, write_oof_csv,
)

RES, FIG, LOGS = HERE / "results", HERE / "figures", HERE / "logs"

# Nearest Specim IQ band to typical camera RGB (same as 06_resnet_transfer.py).
RGB_NM = (640.0, 550.0, 460.0)
RGB_NAMES = ("R", "G", "B")


def rgb_bands(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the closest band to each RGB centre, and the actual centre wavelengths (nm)."""
    idx = np.array([int(np.argmin(np.abs(wavelengths - nm))) for nm in RGB_NM])
    return idx, wavelengths[idx]


def load_features(spectra: str):
    """(n, 3) mean reflectance at R/G/B, integer variety labels, vineyard row, leaf table.

    SNV is over the three RGB values of each leaf (shape of the triplet, not brightness).
    """
    meta = load_meta()
    X, _ = leaf_mean_spectra(meta)
    idx, _ = rgb_bands(load_wavelengths())
    X = X[:, idx]
    if spectra == "snv":
        X = snv(X)
    y = meta.variety.map({v: k for k, v in enumerate(VARIETIES)}).to_numpy()
    rows = meta.row.to_numpy()
    return X, y, rows, meta, idx


def metrics_markdown(spectra: str, metrics: dict, folds: dict, actual_nm: np.ndarray) -> str:
    m, cm = metrics, metrics["confusion"]
    ci = m["balanced_accuracy_ci95"]
    bands = ", ".join(f"{n} {w:.1f} nm" for n, w in zip(RGB_NAMES, actual_nm))
    snv_note = " -> SNV" if spectra == "snv" else ""
    lines = [
        f"# Variety prediction with logistic regression (RGB, {spectra})",
        "",
        f"Leaf mean at three bands ({bands}){snv_note} -> standardise -> multinomial "
        "logistic regression, C chosen inside each training split. Same 5 row folds as the "
        "204-band run; every leaf predicted by the model that never saw its row.",
        "",
        f"**Balanced accuracy: {m['balanced_accuracy']:.3f}** "
        f"(95% CI {ci[0]:.3f}-{ci[1]:.3f}; chance 0.25). Macro-F1: {m['macro_f1']:.3f}.",
        "",
        "Confusion matrix (rows = true variety, columns = predicted):",
        "",
        "| true \\ predicted | " + " | ".join(VARIETIES) + " | total |",
        "|---|" + "---|" * (len(VARIETIES) + 1),
    ]
    lines += [
        f"| **{v}** | " + " | ".join(str(c) for c in cm[k]) + f" | {cm[k].sum()} |"
        for k, v in enumerate(VARIETIES)
    ]
    lines += [
        "",
        "| variety | precision | recall | F1 | n | ROC AUC (one vs rest) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {v} | {m['precision'][k]:.3f} | {m['recall'][k]:.3f} | {m['f1'][k]:.3f} | "
        f"{m['support'][k]} | {m['auc_ovr'][k]:.3f} |"
        for k, v in enumerate(VARIETIES)
    ]
    lines += [
        "",
        "## Per fold (held-out row)",
        "",
        "| test row | chosen C | test leaves | balanced accuracy |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r} | {f['chosen_C']:g} | {f['test_leaves']} | {f['test_balanced_accuracy']:.3f} |"
        for r, f in folds.items()
    ]
    lines += [
        "",
        "Caveat: 'variety' also carries anything acquired together with it (imaging day, lighting, leaf "
        "handling), so a high score shows the spectra separate the varieties, not that cultivar biology alone "
        "is the cause.",
        "",
    ]
    return "\n".join(lines)


def write_metrics_json(path: Path, spectra: str, metrics: dict, folds: dict,
                       idx: np.ndarray, actual_nm: np.ndarray) -> None:
    m = metrics
    payload = {
        "spectra": spectra,
        "features": "rgb",
        "rgb_target_nm": list(RGB_NM),
        "rgb_band_index": idx.tolist(),
        "rgb_actual_nm": actual_nm.tolist(),
        "balanced_accuracy": m["balanced_accuracy"],
        "balanced_accuracy_ci95": list(m["balanced_accuracy_ci95"]),
        "macro_f1": m["macro_f1"],
        "classes": VARIETIES,
        "confusion": m["confusion"].tolist(),
        "per_variety": {
            v: {
                "precision": m["precision"][k],
                "recall": m["recall"][k],
                "f1": m["f1"][k],
                "n": int(m["support"][k]),
                "auc_ovr": m["auc_ovr"][k],
            }
            for k, v in enumerate(VARIETIES)
        },
        "folds": folds,
    }
    path.write_text(json.dumps(payload, indent=2, default=float))


def plot_coefficients(coef: np.ndarray, actual_nm: np.ndarray, C: float, path: Path) -> None:
    """Bar chart: one group per RGB band, one bar per variety (coef shape is 4 varieties × 3 bands)."""
    labels = [f"{n}\n{w:.0f} nm" for n, w in zip(RGB_NAMES, actual_nm)]
    x = np.arange(len(RGB_NAMES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in enumerate(VARIETIES):
        ax.bar(x + (k - 1.5) * width, coef[k], width, label=v)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, labels)
    ax.set(ylabel="weight (per standardised band)",
           title=f"RGB band weights per variety (all leaves, C = {C:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(spectra: str, meta, P: np.ndarray, metrics: dict, folds: dict,
                  coef: np.ndarray, C: float, idx: np.ndarray, actual_nm: np.ndarray) -> str:
    sfx = "" if spectra == "snv" else "_raw"
    write_oof_csv(meta, P, metrics["pred"], RES / f"variety_oof{sfx}.csv")
    md = metrics_markdown(spectra, metrics, folds, actual_nm)
    (RES / f"variety_metrics{sfx}.md").write_text(md)
    write_metrics_json(RES / f"variety_metrics{sfx}.json", spectra, metrics, folds, idx, actual_nm)
    plot_confusion(metrics["confusion"], f"rgb-{spectra}", metrics["balanced_accuracy"],
                   FIG / f"confusion_matrix{sfx}.png")
    plot_coefficients(coef, actual_nm, C, FIG / f"coefficients{sfx}.png")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", choices=["snv", "raw"], default="snv")
    args = ap.parse_args()
    for d in (RES, FIG, LOGS):
        d.mkdir(exist_ok=True)
    log_file = LOGS / ("run.log" if args.spectra == "snv" else "run_raw.log")
    log_file.unlink(missing_ok=True)
    log = get_logger("variety_logistic_regression_rgb", log_file=log_file)

    X, y, rows, meta, idx = load_features(args.spectra)
    _, actual_nm = rgb_bands(load_wavelengths())
    P, folds = nested_row_cv(X, y, rows)
    metrics = evaluate(y, P)
    C, coef = fit_coefficient_model(X, y, folds)
    md = write_outputs(args.spectra, meta, P, metrics, folds, coef, C, idx, actual_nm)
    log.info(md)


if __name__ == "__main__":
    main()
