"""Predict each leaf's variety from its mean spectrum with logistic regression."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, log_loss  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data import load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from pca_common import VARIETIES, get_device, np_, snv  # noqa: E402
from report_tables import BASELINE, BANDS_TITLE, column_table, fold_note, main_table  # noqa: E402
from scoring import evaluate  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
ROWS = ["R1", "R2", "R3", "R4", "R5"]
BANDS = ("all", "rgb")
RGB_NM = {"R": 640.0, "G": 550.0, "B": 460.0}  # the pseudo-RGB picks of the earlier ResNet runs
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
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
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="lbfgs", max_iter=10_000))


def fold_split(rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = row k, validation = row k+1, train = the other 3 rows.

    Every variety model, CNN or logistic regression, uses this one split, so each sees the
    same rows in the same roles. Whole rows are held out because neighbouring vines look alike.

    n: the number of leaves.

    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param k: Fold number 0-4.
    :type k: int
    :return: train, validation, test leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(rows))
    test, val = rows == ROWS[k], rows == ROWS[(k + 1) % 5]
    return idx[~test & ~val], idx[val], idx[test]


def fit_with_validation(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[float, object]:
    """Fit one model per C on the training rows and keep the one with the lowest validation log-loss.

    The validation row only picks C and never fits the model, as it only picks the checkpoint
    in the CNN.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param tr: Training leaves.
    :type tr: np.ndarray
    :param va: Validation leaves.
    :type va: np.ndarray
    :return: The best C and its fitted model.
    :rtype: tuple
    """
    fits = {C: make_model(C).fit(X[tr], y[tr]) for C in C_GRID}
    losses = {C: log_loss(y[va], m.predict_proba(X[va]), labels=range(len(VARIETIES))) for C, m in fits.items()}
    best = min(losses, key=losses.get)
    return best, fits[best]


def row_cv(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hold out one field row at a time, with :func:`fold_split` and :func:`fit_with_validation`.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :return: Out-of-fold probabilities (n, 4) and per-fold {chosen_C, score}, keyed by test row.
    :rtype: tuple
    """
    P = np.empty((len(y), len(VARIETIES)))
    folds = {}
    for k, r in enumerate(ROWS):
        tr, va, te = fold_split(rows, k)
        best, model = fit_with_validation(X, y, tr, va)
        P[te] = model.predict_proba(X[te])
        folds[r] = {"chosen_C": best, "test_balanced_accuracy": balanced_accuracy_score(y[te], P[te].argmax(1))}
    return P, folds


def write_metrics_json(path: Path, cfg: Config, m: dict, folds: dict) -> None:
    """Write metrics.json, which results/metrics.md is built from.

    :param path: Output file.
    :type path: Path
    :param cfg: The configuration that was run.
    :type cfg: Config
    :param m: Output of scoring.evaluate.
    :type m: dict
    :param folds: Per-fold results of :func:`row_cv`.
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


def metrics_markdown(metrics: dict) -> str:
    """Build results/metrics.md in the shared format (report_tables): one section per input.

    :param metrics: {configuration name: parsed metrics.json}.
    :type metrics: dict
    :return: Markdown text.
    :rtype: str
    """
    lines = ["# Variety: logistic regression", "",
             "Leaf mean spectrum → SNV → logistic regression, C chosen on the validation row. Balanced accuracy = "
             "the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval.", ""]
    for name in (b for b in BANDS if b in metrics):
        m = metrics[name]
        folds = m["folds"].values()
        lines += [f"## {BANDS_TITLE[name]}", ""]
        lines += main_table([{"name": BASELINE, "bal_acc": m["balanced_accuracy"],
                              "ci95": m["balanced_accuracy_ci95"],
                              "folds": [f["test_balanced_accuracy"] for f in folds]}])
        lines += ["", fold_note(VARIETIES), ""]
        lines += column_table("Recall per variety:", VARIETIES,
                              {BASELINE: {v: m["per_variety"][v]["recall"] for v in VARIETIES}})
        lines += ["", "Chosen C per fold: " + ", ".join(f"{f['chosen_C']:g}" for f in folds) + ".", ""]
    return "\n".join(lines)


def write_outputs(cfg: Config, m: dict, folds: dict) -> None:
    """Write the configuration's metrics.json and confusion matrix, then rebuild results/metrics.md.

    :param cfg: The configuration that was run.
    :type cfg: Config
    :param m: Output of scoring.evaluate.
    :type m: dict
    :param folds: Per-fold results of :func:`row_cv`.
    :type folds: dict
    """
    res, fig = RES / cfg.name, FIG / cfg.name
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    write_metrics_json(res / "metrics.json", cfg, m, folds)
    plot_confusion(m["confusion"], cfg, m["balanced_accuracy"], fig / "confusion_matrix.png")
    metrics = {p.parent.name: json.loads(p.read_text()) for p in RES.glob("*/metrics.json")}
    (RES / "metrics.md").write_text(metrics_markdown(metrics))


def main() -> None:
    """Parse config → log → load → row CV → metrics → write files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=BANDS, default=Config.bands, help="all 204 bands, or only the 3 RGB bands")
    args = ap.parse_args()
    cfg = Config(args.bands)
    log = get_logger("variety_logreg")

    device = get_device()
    d = load_features(cfg, device)
    P, folds = row_cv(d["X"], d["y"], d["rows"])
    m = evaluate(d["y"], P, device)
    write_outputs(cfg, m, folds)
    log.info("%s bands: balanced accuracy %.3f | results/metrics.md", cfg.name, m["balanced_accuracy"])


if __name__ == "__main__":
    main()
