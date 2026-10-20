"""Predict the variety of every leaf (Baco / Cab Franc / Chardonnay / Vidal) with logistic regression.

Run from claude runs/variety_logistic_regression/:
    ../../.venv/bin/python -u variety_logistic_regression.py
    ../../.venv/bin/python -u variety_logistic_regression.py --spectra raw     # without SNV

Pipeline
  1. Leaf spectrum = the mean over all of the leaf's pixels, 204 bands (../shared/spectra.py).
  2. SNV (default): each leaf's spectrum centred and scaled over its own bands, which removes overall
     brightness, so the model has to use spectral shape; --spectra raw skips it.
  3. Per-band standardisation (mean 0, SD 1), fitted on the training leaves only.
  4. Multinomial logistic regression (one probability per variety). Its regularisation C is picked
     inside each training split: leave-one-row-out over the 4 training rows, scored by log-loss,
     over C = 1e-5 ... 1000. The test row never influences C.
Evaluation: 5 folds = rows R1-R5 (all varieties' leaves of one row held out together, as in every
other pipeline in this repo: neighbouring vines look alike, so a random split would leak). Every leaf
is predicted once, by the model that never saw its row. C is chosen by the inner CV ("nested" CV)
because picking it on the test rows would let them tune the model and inflate the score.

Outputs (results/, suffix _raw when --spectra raw)
  variety_oof.csv        per leaf: true variety, row, the 4 probabilities, predicted variety
  variety_metrics.md     confusion matrix, balanced accuracy (leaf-bootstrap 95% CI), macro-F1,
                         per-variety precision / recall / F1 / one-vs-rest ROC AUC, per-fold results
  variety_metrics.json   the same numbers, plus the inner-CV log-loss of every C per fold
  figures/confusion_matrix.png
  figures/coefficients.png   the model's weight on every band per variety (fit on all leaves,
                             C = the most often chosen value), to show which wavelengths it uses
  logs/run.log (logs/run_raw.log)   what the run printed (the metrics markdown)

Why not torch: the reported numbers come from scikit-learn's LogisticRegression (lbfgs) and the
bootstrap CI from the numpy RNG stream; swapping libraries would change them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))                     # common.py, spectra.py, logs.py
from common import load_meta, load_wavelengths                       # noqa: E402
from logs import get_logger                                          # noqa: E402
from spectra import leaf_mean_spectra, snv                           # noqa: E402

RES, FIG, LOGS = HERE / "results", HERE / "figures", HERE / "logs"
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
N_BOOT, SEED = 2000, 0


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(C: float):
    """Scaler + multinomial logistic regression.

    StandardScaler is inside the pipeline so it is always fit on the training
    leaves of the current split, never on the held-out row. C is L2 inverse
    regularisation: small C = a simpler (more regularised) model.
    sklearn, not torch: this is the reported model.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10_000))


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def load_features(spectra: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """One mean spectrum per leaf, integer variety labels, and the vineyard row of each leaf.

    spectra='snv'  — subtract each leaf's own mean and divide by its own SD (shape only).
    spectra='raw'  — keep brightness; useful as a comparison, not the default.
    y is 0..3 in VARIETIES order. rows are strings like 'R1'..'R5', the outer CV folds.
    """
    meta = load_meta()
    X, _ = leaf_mean_spectra(meta)
    if spectra == "snv":
        X = snv(X)
    y = meta.variety.map({v: k for k, v in enumerate(VARIETIES)}).to_numpy()
    rows = meta.row.to_numpy()
    return X, y, rows, meta


# --------------------------------------------------------------------------
# Cross-validation (no I/O)
# --------------------------------------------------------------------------
def leave_one_row_out_proba(X: np.ndarray, y: np.ndarray, rows: np.ndarray, C: float) -> np.ndarray:
    """Predict every leaf with a model that never trained on that leaf's row, for one C.

    Used as the *inner* loop: among the 4 training rows, hold one out, score log-loss,
    pick C. The outer test row is already stripped off before this is called.
    """
    P = np.empty((len(y), len(VARIETIES)))
    for r in np.unique(rows):
        te = rows == r
        P[te] = make_model(C).fit(X[~te], y[~te]).predict_proba(X[te])
    return P


def choose_C(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[float, dict]:
    """C with the lowest leave-one-row-out log-loss on this training split.

    Returns that C and the log-loss of every candidate (for the metrics JSON).
    """
    scores = {}
    for C in C_GRID:
        P = leave_one_row_out_proba(X, y, rows, C)
        scores[C] = log_loss(y, P, labels=range(len(VARIETIES)))
    best = min(scores, key=scores.get)
    return best, scores


def nested_row_cv(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Outer 5-fold evaluation: one vineyard row held out at a time.

    For each held-out row:
      1. Tune C on the other 4 rows (inner leave-one-row-out log-loss).
      2. Refit that C on all 4 training rows.
      3. Predict the held-out row.

    Every leaf therefore gets one out-of-fold probability vector from a model
    that never saw its row, and never saw its row when C was chosen either.
    """
    P = np.empty((len(y), len(VARIETIES)))
    folds = {}
    for r in np.unique(rows):
        te = rows == r
        best_C, inner = choose_C(X[~te], y[~te], rows[~te])
        P[te] = make_model(best_C).fit(X[~te], y[~te]).predict_proba(X[te])
        folds[r] = {
            "chosen_C": best_C,
            "inner_log_loss": {str(c): v for c, v in inner.items()},
            "test_leaves": int(te.sum()),
        }
    pred = P.argmax(1)
    for r, rec in folds.items():
        rec["test_balanced_accuracy"] = balanced_accuracy_score(y[rows == r], pred[rows == r])
    return P, folds


# --------------------------------------------------------------------------
# Metrics (no I/O). Balanced accuracy = mean recall over the 4 varieties, so the large Cab Franc
# and Vidal groups do not dominate; chance = 0.25.
# --------------------------------------------------------------------------
def bootstrap_balanced_accuracy(y: np.ndarray, pred: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Leaf-level 95% interval: resample leaves with replacement, recompute balanced accuracy.
    numpy rng, not torch: its stream gives the reported CI."""
    rng = np.random.default_rng(seed)
    boots = [
        balanced_accuracy_score(y[i], pred[i])
        for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
    ]
    return np.percentile(boots, [2.5, 97.5])


def evaluate(y: np.ndarray, P: np.ndarray) -> dict:
    """Confusion matrix, balanced accuracy, per-variety precision/recall/F1/AUC from OOF probs."""
    pred = P.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=range(len(VARIETIES)), zero_division=0
    )
    return {
        "pred": pred,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "balanced_accuracy_ci95": bootstrap_balanced_accuracy(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": sup,
        # One-vs-rest AUC: how well P(this variety) ranks those leaves above the other three.
        "auc_ovr": [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(len(VARIETIES))],
        "confusion": confusion_matrix(y, pred, labels=range(len(VARIETIES))),
    }


def fit_coefficient_model(X: np.ndarray, y: np.ndarray, folds: dict):
    """Refit on *all* leaves so the coefficient plot can show which bands the model uses.

    C is the value chosen most often in the outer folds. This fit is for interpretation
    only; it is not used for the reported accuracy (that comes from nested_row_cv).
    """
    c_common = Counter(f["chosen_C"] for f in folds.values()).most_common(1)[0][0]
    fitted = make_model(c_common).fit(X, y)
    return c_common, fitted[-1].coef_


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def write_oof_csv(meta, P: np.ndarray, pred: np.ndarray, path: Path) -> None:
    """Per leaf: path, true variety, row, the 4 out-of-fold probabilities and the predicted variety."""
    oof = meta[["path", "variety", "row"]].copy()
    for k, v in enumerate(VARIETIES):
        oof[f"p_{v}"] = P[:, k]
    oof["predicted"] = [VARIETIES[k] for k in pred]
    oof.to_csv(path, index=False)


def metrics_markdown(spectra: str, metrics: dict, folds: dict) -> str:
    """results/variety_metrics*.md: headline, confusion matrix, per-variety and per-fold tables."""
    m, cm = metrics, metrics["confusion"]
    ci = m["balanced_accuracy_ci95"]
    snv_note = " -> SNV" if spectra == "snv" else ""
    lines = [
        f"# Variety prediction with logistic regression ({spectra} spectra)",
        "",
        f"Leaf mean spectrum{snv_note} -> standardise -> multinomial "
        "logistic regression, C chosen inside each training split. 5 row folds; every leaf predicted by the "
        "model that never saw its row.",
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


def write_metrics_json(path: Path, spectra: str, metrics: dict, folds: dict) -> None:
    """results/variety_metrics*.json: the markdown's numbers plus the inner-CV log-loss of every C."""
    m = metrics
    payload = {
        "spectra": spectra,
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


def plot_confusion(cm: np.ndarray, spectra: str, bacc: float, path: Path) -> None:
    """figures/confusion_matrix*.png: (4, 4) counts, rows = true variety."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(VARIETIES)):
        for j in range(len(VARIETIES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(VARIETIES)), VARIETIES, rotation=20)
    ax.set_yticks(range(len(VARIETIES)), VARIETIES)
    ax.set(xlabel="predicted", ylabel="true",
           title=f"Variety prediction ({spectra}): balanced acc. {bacc:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coefficients(coef: np.ndarray, wavelengths: np.ndarray, C: float, path: Path) -> None:
    """figures/coefficients*.png: (4, 204) weights per standardised band, one line per variety."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, v in enumerate(VARIETIES):
        ax.plot(wavelengths, coef[k], lw=1.6, label=v)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="wavelength (nm)", ylabel="weight (per standardised band)",
           xlim=(wavelengths[0], wavelengths[-1]),
           title=f"Which bands the model uses: weight per variety (all leaves, C = {C:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(spectra: str, meta, P: np.ndarray, metrics: dict, folds: dict,
                  coef: np.ndarray, C: float) -> str:
    """Write CSV / markdown / JSON / figures. Returns the markdown so main can log it."""
    sfx = "" if spectra == "snv" else "_raw"
    write_oof_csv(meta, P, metrics["pred"], RES / f"variety_oof{sfx}.csv")
    md = metrics_markdown(spectra, metrics, folds)
    (RES / f"variety_metrics{sfx}.md").write_text(md)
    write_metrics_json(RES / f"variety_metrics{sfx}.json", spectra, metrics, folds)
    plot_confusion(metrics["confusion"], spectra, metrics["balanced_accuracy"],
                   FIG / f"confusion_matrix{sfx}.png")
    plot_coefficients(coef, load_wavelengths(), C, FIG / f"coefficients{sfx}.png")
    return md


# --------------------------------------------------------------------------
def main():
    """Parse --spectra -> set up logs/run[_raw].log -> load -> nested row CV -> metrics -> write outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", choices=["snv", "raw"], default="snv")
    args = ap.parse_args()
    for d in (RES, FIG, LOGS):
        d.mkdir(exist_ok=True)
    log_file = LOGS / ("run.log" if args.spectra == "snv" else "run_raw.log")
    log_file.unlink(missing_ok=True)                               # the log holds the latest run only
    log = get_logger("variety_logistic_regression", log_file=log_file)

    X, y, rows, meta = load_features(args.spectra)
    P, folds = nested_row_cv(X, y, rows)
    metrics = evaluate(y, P)
    C, coef = fit_coefficient_model(X, y, folds)
    md = write_outputs(args.spectra, meta, P, metrics, folds, coef, C)
    log.info(md)


if __name__ == "__main__":
    main()
