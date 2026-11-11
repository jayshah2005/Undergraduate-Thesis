"""Score the variety CNN configurations against each other and against logistic regression."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiments import RESULTS, ROWS, SEEDS, VARIETIES, run_name  # noqa: E402  (also sets sys.path)
from helpers.data import load_leaf_sizes  # noqa: E402
from logistic_regression import (BANDS, INK, MUTED, get_device, get_logger, load_features,  # noqa: E402
                                 metrics_section, plot_confusion, row_cv)
from logistic_regression import Config as LRConfig  # noqa: E402
from helpers.report_tables import (BANDS_TITLE, BASELINE, METRICS_INTRO, column_table, fold_note,  # noqa: E402
                           main_table, seed_table)
from scoring import N_BOOT, SEED, evaluate  # noqa: E402

FIG = HERE / "figures"
BAND_COLOR = {"all": "#2a78d6", "rgb": "#eb6834"}
GRID = "#e1e0d9"


def load_ensemble(run: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect one run's test probabilities and average them over the seeds.

    n: the number of leaves. s: the number of seeds.

    :param run: Results folder name, all or rgb.
    :type run: str
    :param n: Number of leaves.
    :type n: int
    :return: Ensemble probabilities (n, 4) and per-seed probabilities (s, n, 4).
    :rtype: tuple
    """
    P = np.full((len(SEEDS), n, 4), np.nan)
    for s in SEEDS:
        for k in range(5):
            f = np.load(RESULTS / run / f"seed{s}" / f"fold{k}.npz")
            P[s, f["test_idx"]] = f["prob"]
    assert not np.isnan(P).any(), f"missing folds for {run}"
    return P.mean(0), P


def leaf_size_control(device: torch.device) -> float:
    """Score logistic regression on leaf size alone: how far could leaf size carry a CNN?

    n: the number of leaves.

    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: Balanced accuracy.
    :rtype: float
    """
    meta, sizes = load_leaf_sizes()
    y = meta.variety.map({v: k for k, v in enumerate(VARIETIES)}).to_numpy()
    P, _folds = row_cv(sizes, y, meta.row.to_numpy().astype(str))
    return evaluate(y, P, device)["balanced_accuracy"]


def paired_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, device: torch.device) -> list[float]:
    """Balanced accuracy of A minus B with a 95% interval, on the same resampled leaves.

    n: the number of leaves.

    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param pred_a: Predictions of A, length n.
    :type pred_a: np.ndarray
    :param pred_b: Predictions of B, length n.
    :type pred_b: np.ndarray
    :param device: Where the resampling runs.
    :type device: torch.device
    :return: Difference, lower bound, upper bound.
    :rtype: list[float]
    """
    yt = torch.tensor(y, device=device)
    g = torch.Generator(device=device).manual_seed(SEED)
    idx = torch.randint(len(y), (N_BOOT, len(y)), generator=g, device=device)
    y_boot = yt[idx]

    def bacc(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Mean recall over the 4 varieties, along the last dimension of truth."""
        hit = pred == truth
        return torch.stack([((truth == k) & hit).sum(-1) / (truth == k).sum(-1) for k in range(4)]).mean(0).double()

    a, b = torch.tensor(pred_a, device=device), torch.tensor(pred_b, device=device)
    diff = bacc(a[idx], y_boot) - bacc(b[idx], y_boot)
    q = torch.quantile(diff, torch.tensor([0.025, 0.975], device=device, dtype=torch.float64)).tolist()
    return [(bacc(a, yt) - bacc(b, yt)).item(), *q]


def score_run(run: str, y: np.ndarray, rows: np.ndarray, device: torch.device) -> dict:
    """Score one run: its 3-seed ensemble, each seed and each row.

    n: the number of leaves.

    :param run: Results folder name, all or rgb.
    :type run: str
    :param y: True variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: Output of evaluate, plus seed_scores, per_row and median_best_epoch.
    :rtype: dict
    """
    ens, seeds = load_ensemble(run, len(y))
    m = evaluate(y, ens, device)
    m["seed_scores"] = [evaluate(y, p, device)["balanced_accuracy"] for p in seeds]
    m["per_row"] = {r: balanced_accuracy_score(y[rows == r], ens[rows == r].argmax(1)) for r in ROWS}
    m["median_best_epoch"] = float(np.median([np.load(RESULTS / run / f"seed{s}" / f"fold{k}.npz")["best_epoch"]
                                              for s in SEEDS for k in range(5)]))
    return m


def compute(device: torch.device, augment: bool = True) -> dict:
    """Score the CNN on each input and its logistic-regression partner, with paired differences.

    :param device: Where the bootstrap runs.
    :type device: torch.device
    :param augment: Score the CNN runs trained with rotations and flips (the default), or those without.
    :type augment: bool
    :return: {"cnn": per input, "lr": per input, "vs_lr": CNN − LR per input, "deltas": 204 bands − RGB}.
    :rtype: dict
    """
    out = {"cnn": {}, "lr": {}}
    for b in BANDS:
        d = load_features(LRConfig(b), device)
        y, rows = d["y"], d["rows"]
        P, folds = row_cv(d["X"], y, rows)
        out["lr"][b] = {**evaluate(y, P, device), "per_row": {r: f["test_balanced_accuracy"] for r, f in folds.items()}}
        out["cnn"][b] = score_run(run_name(b, augment), y, rows, device)
    cnn, lr = out["cnn"], out["lr"]
    out["vs_lr"] = {b: paired_delta(y, cnn[b]["pred"], lr[b]["pred"], device) for b in BANDS}
    out["deltas"] = {
        "CNN, mean pool": paired_delta(y, cnn["all"]["pred"], cnn["rgb"]["pred"], device),
        BASELINE: paired_delta(y, lr["all"]["pred"], lr["rgb"]["pred"], device),
    }
    return out


def to_markdown(r: dict, b: str, augment: bool = True) -> str:
    """Build one run's final_report.md in the shared format (report_tables): balanced accuracy.

    :param r: Output of :func:`compute`.
    :type r: dict
    :param b: The run's input, a value of BANDS.
    :type b: str
    :param augment: Whether the CNN runs were trained with rotations and flips.
    :type augment: bool
    :return: Markdown text.
    :rtype: str
    """
    def model(name: str, m: dict, vs_lr: list[float] | None = None) -> dict:
        """One line of the main table."""
        line = {"name": name, "bal_acc": m["balanced_accuracy"], "ci95": m["balanced_accuracy_ci95"],
                "folds": [m["per_row"][x] for x in ROWS]}
        if vs_lr is not None:
            line["delta"], line["delta_ci95"] = vs_lr[0], vs_lr[1:]
        return line

    cnn, lr = r["cnn"][b], r["lr"][b]
    lines = [f"# Variety: whole-leaf CNN, {BANDS_TITLE[b]}" + ("" if augment else ", trained without augmentation"),
             "", "Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every "
             "model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, "
             "logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic "
             "regression on the same input. Every metric: `metrics.md`.", "", f"## {BANDS_TITLE[b]}", ""]
    lines += main_table([model(BASELINE, lr), model("CNN, mean pool", cnn, r["vs_lr"][b])])
    lines += ["", fold_note(VARIETIES), ""]
    lines += column_table("Recall per variety:", VARIETIES,
                          {BASELINE: dict(zip(VARIETIES, lr["recall"])),
                           "CNN, mean pool": dict(zip(VARIETIES, cnn["recall"]))})
    lines += [""] + seed_table({"CNN, mean pool": cnn["seed_scores"]})
    lines += ["", f"Median best epoch: {cnn['median_best_epoch']:.0f}.", ""]
    # The same comparison of inputs goes in both runs' reports, so each report stands alone.
    lines += ["## 204 bands − RGB (same resampled leaves)", "", "| model | Δ balanced accuracy [95% CI] |", "|---|---|"]
    lines += [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in r["deltas"].items()]
    lines += ["", f"Leaf size alone (pixel count, height, width), logistic regression: {r['leaf_size']:.3f}.", ""]
    return "\n".join(lines)


def metrics_markdown(r: dict, b: str, augment: bool = True) -> str:
    """Build one run's metrics.md: every metric of both models.

    :param r: Output of :func:`compute`.
    :type r: dict
    :param b: The run's input, a value of BANDS.
    :type b: str
    :param augment: Whether the CNN runs were trained with rotations and flips.
    :type augment: bool
    :return: Markdown text.
    :rtype: str
    """
    lines = [f"# Variety: whole-leaf CNN, {BANDS_TITLE[b]}: all metrics"
             + ("" if augment else ", trained without augmentation"), "",
             "4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap "
             "intervals. " + METRICS_INTRO, "", f"## {BANDS_TITLE[b]}", ""]
    return "\n".join(lines + metrics_section({BASELINE: r["lr"][b], "CNN, mean pool": r["cnn"][b]}) + [""])


def plot_scores(r: dict, path: Path) -> None:
    """Dot plot: each model's balanced accuracy with its interval, coloured by input bands.

    :param r: Output of :func:`compute`.
    :type r: dict
    :param path: Output file.
    :type path: Path
    """
    # Each row is (label, 204-band result, RGB result).
    groups = [("CNN", r["cnn"]["all"], r["cnn"]["rgb"]),
              ("Logistic regression", r["lr"]["all"], r["lr"]["rgb"])]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for i, (_label, a, b) in enumerate(groups):
        for off, m, bands in ((-0.14, a, "all"), (0.14, b, "rgb")):
            lo, hi = m["balanced_accuracy_ci95"]
            ax.plot([lo, hi], [i + off] * 2, color=BAND_COLOR[bands], lw=2, solid_capstyle="round")
            ax.plot(m["balanced_accuracy"], i + off, "o", ms=8, color=BAND_COLOR[bands], mec="white", mew=1.5,
                    label={"all": "204 bands", "rgb": "RGB (3 bands)"}[bands] if i == 0 else None)
    ax.axvline(0.25, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("balanced accuracy, held-out rows (dashed = chance)", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_outputs(r: dict, augment: bool = True) -> str:
    """Write each run's final_report.md and metrics.md, and the figures (_no_augmentation suffix for those runs).

    :param r: Output of :func:`compute`.
    :type r: dict
    :param augment: Whether the CNN runs were trained with rotations and flips.
    :type augment: bool
    :return: Where the reports went.
    :rtype: str
    """
    suffix = "" if augment else "_no_augmentation"
    FIG.mkdir(exist_ok=True)
    for b in BANDS:
        run = RESULTS / run_name(b, augment)
        (run / "final_report.md").write_text(to_markdown(r, b, augment))
        (run / "metrics.md").write_text(metrics_markdown(r, b, augment))
    plot_scores(r, FIG / f"scores{suffix}.png")
    for name, m in r["cnn"].items():
        plot_confusion(m["confusion"], SimpleNamespace(name=name), m["balanced_accuracy"],
                       FIG / f"confusion_{name}{suffix}.png")
    return "{" + ",".join(run_name(b, augment) for b in BANDS) + "}/{final_report,metrics}.md"


def main() -> None:
    """Load every run → metrics → write files."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augmentation", action="store_true",
                    help="report the runs from run_experiments.py --no-augmentation")
    args = ap.parse_args()
    log = get_logger("variety_cnn_report")
    device = get_device()
    augment = not args.no_augmentation
    md = write_outputs({**compute(device, augment), "leaf_size": leaf_size_control(device)}, augment)
    log.info("[saved] results/%s, figures/", md)


if __name__ == "__main__":
    main()
