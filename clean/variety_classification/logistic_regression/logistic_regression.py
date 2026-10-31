"""Predict each leaf's variety from its mean spectrum with logistic regression."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, log_loss, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data import load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from pca_common import VAR_COLORS, VARIETIES, get_device, np_, snv  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
BANDS = ("all", "rgb")
RGB_NM = {"R": 640.0, "G": 550.0, "B": 460.0}  # the pseudo-RGB picks of the earlier ResNet runs
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
N_BOOT, SEED = 2000, 0
INK, MUTED = "#0b0b0b", "#52514e"


@dataclass(frozen=True)
class Config:
    """Which numbers describe a leaf; SNV is always applied, over the bands kept."""

    bands: str = "all"    # "all": every band; "rgb": the bands nearest 640 / 550 / 460 nm

    @property
    def name(self) -> str:
        """Folder name of this configuration."""
        return self.bands


def rgb_bands(wavelengths: np.ndarray) -> np.ndarray:
    """Find the band closest to each of the three RGB centres.

    :param wavelengths: Band centres in nm, shape (204,).
    :type wavelengths: np.ndarray
    :return: Band indices in R, G, B order, length 3.
    :rtype: np.ndarray
    """
    return np.array([int(np.abs(wavelengths - nm).argmin()) for nm in RGB_NM.values()])


def load_features(cfg: Config, device: torch.device) -> dict:
    """Build the feature table of one configuration.

    SNV centres and scales each leaf over its own bands, which removes brightness. Bands
    are picked before SNV, so under rgb it works over the three values only.

    n: the number of leaves (998). d: the number of features (204 or 3).

    :param cfg: Which bands to use.
    :type cfg: Config
    :param device: Where SNV runs.
    :type device: torch.device
    :return: X (n, d) float64, y (n,) with codes in VARIETIES order, rows (n,), meta (n rows),
        wavelengths of the kept bands (d,).
    :rtype: dict
    """
    meta, wavelengths, means, _glint = load_dataset()
    keep = rgb_bands(wavelengths) if cfg.bands == "rgb" else np.arange(len(wavelengths))
    # Why NumPy afterwards: scikit-learn fits the model.
    X = np_(snv(torch.tensor(means[:, keep], dtype=torch.float64, device=device)))
    y = meta.variety.map({v: k for k, v in enumerate(VARIETIES)}).to_numpy()
    return {"X": X, "y": y, "rows": meta.row.to_numpy(), "meta": meta, "wavelengths": wavelengths[keep]}


def make_model(C: float):
    """Standardise each band, then fit multinomial logistic regression.

    Why scikit-learn: it is the model behind the earlier reported score, and its lbfgs
    solver has no drop-in PyTorch equivalent. The scaler is inside the pipeline so it is
    only ever fit on the training leaves of the current split.

    :param C: Inverse L2 regularisation; small C = a simpler model.
    :type C: float
    :return: Unfitted scikit-learn pipeline.
    :rtype: sklearn.pipeline.Pipeline
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10_000))


def choose_C(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> float:
    """Pick the C with the lowest leave-one-row-out log-loss on the training leaves.

    k: the number of training leaves. d: the number of features.

    :param X: Training features, shape (k, d).
    :type X: np.ndarray
    :param y: Variety codes, length k.
    :type y: np.ndarray
    :param rows: Field row per leaf, length k.
    :type rows: np.ndarray
    :return: The best C.
    :rtype: float
    """
    scores = {}
    for C in C_GRID:
        P = np.empty((len(y), len(VARIETIES)))
        for r in np.unique(rows):
            te = rows == r
            P[te] = make_model(C).fit(X[~te], y[~te]).predict_proba(X[te])
        scores[C] = log_loss(y, P, labels=range(len(VARIETIES)))
    return min(scores, key=scores.get)


def nested_row_cv(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hold out one field row at a time, tuning C on the other four.

    Whole rows are held out because neighbouring vines look alike. C is tuned inside the
    training rows so the test row can't influence it.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :return: Out-of-fold probabilities (n, 4) and per-row {chosen_C, score}.
    :rtype: tuple
    """
    P = np.empty((len(y), len(VARIETIES)))
    folds = {}
    for r in np.unique(rows):
        te = rows == r
        best = choose_C(X[~te], y[~te], rows[~te])
        P[te] = make_model(best).fit(X[~te], y[~te]).predict_proba(X[te])
        folds[str(r)] = {
            "chosen_C": best,
            "test_balanced_accuracy": balanced_accuracy_score(y[te], P[te].argmax(1)),
        }
    return P, folds


def bootstrap_balanced_accuracy(y: np.ndarray, pred: np.ndarray, device: torch.device) -> list[float]:
    """Compute a 95% interval by resampling leaves with replacement.

    B: the number of resamples. n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param pred: Predicted variety codes, length n.
    :type pred: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: Lower and upper bound.
    :rtype: list[float]
    """
    g = torch.Generator(device=device).manual_seed(SEED)
    yt, pt = torch.tensor(y, device=device), torch.tensor(pred, device=device)
    idx = torch.randint(len(y), (N_BOOT, len(y)), generator=g, device=device)  # (B, n)
    y_boot, hit = yt[idx], (pt == yt)[idx]
    # Balanced accuracy = mean recall over the 4 varieties.
    recalls = torch.stack([((y_boot == k) & hit).sum(1) / (y_boot == k).sum(1)
                           for k in range(len(VARIETIES))])
    return torch.quantile(recalls.mean(0).double(), torch.tensor([0.025, 0.975], device=device,
                                                                dtype=torch.float64)).tolist()


def evaluate(y: np.ndarray, P: np.ndarray, device: torch.device) -> dict:
    """Score out-of-fold probabilities.

    n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param P: Out-of-fold probabilities, shape (n, 4).
    :type P: np.ndarray
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: pred (n,), balanced accuracy and its CI, per-variety recall and ROC AUC, confusion (4, 4).
    :rtype: dict
    """
    pred = P.argmax(1)
    labels = range(len(VARIETIES))
    return {
        "pred": pred,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "balanced_accuracy_ci95": bootstrap_balanced_accuracy(y, pred, device),
        "recall": recall_score(y, pred, labels=labels, average=None),
        # One-vs-rest AUC: how well P(this variety) ranks its leaves above the other three.
        "auc_ovr": [roc_auc_score((y == k).astype(int), P[:, k]) for k in labels],
        "confusion": confusion_matrix(y, pred, labels=labels),
    }


def fit_coefficients(X: np.ndarray, y: np.ndarray, folds: dict) -> tuple[float, np.ndarray]:
    """Refit on all leaves to show which bands the model uses; never used for the score.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param folds: Per-row results of :func:`nested_row_cv`; gives the most often chosen C.
    :type folds: dict
    :return: That C and the weights, shape (4, d).
    :rtype: tuple
    """
    C = Counter(f["chosen_C"] for f in folds.values()).most_common(1)[0][0]
    return C, make_model(C).fit(X, y)[-1].coef_


def metrics_markdown(cfg: Config, m: dict, folds: dict) -> str:
    """Build metrics.md: balanced accuracy, confusion matrix, per-variety and per-row scores.

    :param cfg: The configuration that was run.
    :type cfg: Config
    :param m: Output of :func:`evaluate`.
    :type m: dict
    :param folds: Per-row results of :func:`nested_row_cv`.
    :type folds: dict
    :return: Markdown text.
    :rtype: str
    """
    cm, ci = m["confusion"], m["balanced_accuracy_ci95"]
    lines = [f"# Variety logistic regression: `{cfg.name}` bands", "",
             f"**Balanced accuracy: {m['balanced_accuracy']:.3f}** (95% CI {ci[0]:.3f}-{ci[1]:.3f}; chance 0.25)", "",
             "| true \\ predicted | " + " | ".join(VARIETIES) + " |", "|---|" + "---|" * len(VARIETIES)]
    lines += [f"| **{v}** | " + " | ".join(str(c) for c in cm[k]) + " |" for k, v in enumerate(VARIETIES)]
    lines += ["", "| variety | recall | ROC AUC (one vs rest) |", "|---|---|---|"]
    lines += [f"| {v} | {m['recall'][k]:.3f} | {m['auc_ovr'][k]:.3f} |" for k, v in enumerate(VARIETIES)]
    lines += ["", "| test row | chosen C | balanced accuracy |", "|---|---|---|"]
    lines += [f"| {r} | {f['chosen_C']:g} | {f['test_balanced_accuracy']:.3f} |" for r, f in folds.items()]
    return "\n".join(lines) + "\n"


def write_metrics_json(path: Path, cfg: Config, m: dict, folds: dict) -> None:
    """Write metrics.json, which the comparison table reads.

    :param path: Output file.
    :type path: Path
    :param cfg: The configuration that was run.
    :type cfg: Config
    :param m: Output of :func:`evaluate`.
    :type m: dict
    :param folds: Per-row results of :func:`nested_row_cv`.
    :type folds: dict
    """
    payload = {
        **asdict(cfg), "classes": VARIETIES,
        "balanced_accuracy": m["balanced_accuracy"], "balanced_accuracy_ci95": m["balanced_accuracy_ci95"],
        "confusion": m["confusion"].tolist(),
        "per_variety": {v: {"recall": m["recall"][k], "auc_ovr": m["auc_ovr"][k]} for k, v in enumerate(VARIETIES)},
        "folds": folds,
    }
    path.write_text(json.dumps(payload, indent=2, default=float))


def _style(ax) -> None:
    """Quiet axes: no top or right spine, muted ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_confusion(cm: np.ndarray, cfg: Config, bacc: float, path: Path) -> None:
    """Draw the confusion matrix, rows = true variety; darker = more leaves.

    :param cm: Counts, shape (4, 4).
    :type cm: np.ndarray
    :param cfg: The configuration that was run.
    :type cfg: Config
    :param bacc: Balanced accuracy for the title.
    :type bacc: float
    :param path: Output file.
    :type path: Path
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(VARIETIES)):
        for j in range(len(VARIETIES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(len(VARIETIES)), VARIETIES, rotation=20)
    ax.set_yticks(range(len(VARIETIES)), VARIETIES)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true", color=MUTED)
    ax.set_title(f"{cfg.name}: balanced accuracy {bacc:.3f}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coefficients(coef: np.ndarray, wavelengths: np.ndarray, cfg: Config, C: float, path: Path) -> None:
    """Draw the model's weight on each band per variety: lines for all bands, bars for rgb.

    :param coef: Weights per standardised band, shape (4, d).
    :type coef: np.ndarray
    :param wavelengths: Band centres of the features, length d.
    :type wavelengths: np.ndarray
    :param cfg: The configuration that was run.
    :type cfg: Config
    :param C: The C of the fit, for the title.
    :type C: float
    :param path: Output file.
    :type path: Path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5) if cfg.bands == "rgb" else (11, 4.5))
    if cfg.bands == "rgb":
        width = 0.18
        for k, v in enumerate(VARIETIES):
            ax.bar(np.arange(3) + (k - 1.5) * width, coef[k], width * 0.9, label=v, color=VAR_COLORS[v])
        ax.set_xticks(range(3), [f"{n}\n{w:.0f} nm" for n, w in zip(RGB_NM, wavelengths)])
    else:
        for k, v in enumerate(VARIETIES):
            ax.plot(wavelengths, coef[k], lw=2, label=v, color=VAR_COLORS[v])
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_xlabel("wavelength (nm)", color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("weight per standardised band", color=MUTED)
    ax.set_title(f"{cfg.name}: weight per variety (all leaves, C = {C:g})", loc="left", color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_markdown(metrics: dict) -> str:
    """Build comparison.md: one line per configuration.

    :param metrics: {configuration name: parsed metrics.json}.
    :type metrics: dict
    :return: Markdown text.
    :rtype: str
    """
    lines = ["# Variety prediction: configurations compared", "",
             "Balanced accuracy = the mean of the 4 recalls shown here; chance is 0.25. Every score comes from the "
             "same 5 held-out field rows.", "",
             "| configuration | balanced accuracy (95% CI) | " + " | ".join(f"{v} recall" for v in VARIETIES)
             + " | per-row range |", "|---|---|" + "---|" * (len(VARIETIES) + 1)]
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["balanced_accuracy"]):
        lo, hi = m["balanced_accuracy_ci95"]
        per_row = [f["test_balanced_accuracy"] for f in m["folds"].values()]
        recalls = " | ".join(f"{m['per_variety'][v]['recall']:.3f}" for v in VARIETIES)
        lines.append(f"| `{name}` | {m['balanced_accuracy']:.3f} ({lo:.3f}–{hi:.3f}) | {recalls} | "
                     f"{min(per_row):.2f}–{max(per_row):.2f} |")
    return "\n".join(lines) + "\n"


def write_outputs(cfg: Config, d: dict, m: dict, folds: dict, C: float, coef: np.ndarray) -> None:
    """Write the configuration's tables and figures.

    :param cfg: The configuration that was run.
    :type cfg: Config
    :param d: Output of :func:`load_features`.
    :type d: dict
    :param m: Output of :func:`evaluate`.
    :type m: dict
    :param folds: Per-row results of :func:`nested_row_cv`.
    :type folds: dict
    :param C: C of the coefficient fit.
    :type C: float
    :param coef: Weights, shape (4, d).
    :type coef: np.ndarray
    """
    res, fig = RES / cfg.name, FIG / cfg.name
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    (res / "metrics.md").write_text(metrics_markdown(cfg, m, folds))
    write_metrics_json(res / "metrics.json", cfg, m, folds)
    plot_confusion(m["confusion"], cfg, m["balanced_accuracy"], fig / "confusion_matrix.png")
    plot_coefficients(coef, d["wavelengths"], cfg, C, fig / "coefficients.png")


def main() -> None:
    """Parse config → log → load → nested row CV → metrics → write files, or compare finished runs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=BANDS, default=Config.bands, help="all 204 bands, or only the 3 RGB bands")
    ap.add_argument("--compare", action="store_true", help="table of every finished configuration, no fitting")
    args = ap.parse_args()

    if args.compare:
        metrics = {p.parent.name: json.loads(p.read_text()) for p in RES.glob("*/metrics.json")}
        md = comparison_markdown(metrics)
        (RES / "comparison.md").write_text(md)
        get_logger("variety_logreg_compare").info(md)
        return

    cfg = Config(args.bands)
    log = get_logger("variety_logreg")

    device = get_device()
    d = load_features(cfg, device)
    P, folds = nested_row_cv(d["X"], d["y"], d["rows"])
    m = evaluate(d["y"], P, device)
    C, coef = fit_coefficients(d["X"], d["y"], folds)
    write_outputs(cfg, d, m, folds, C, coef)
    log.info("%s bands: balanced accuracy %.3f | results/%s/metrics.md", cfg.name, m["balanced_accuracy"], cfg.name)


if __name__ == "__main__":
    main()
