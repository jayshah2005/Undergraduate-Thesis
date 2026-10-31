"""Score the variety CNN configurations against each other and against logistic regression."""
from __future__ import annotations

import json
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
from run_experiments import RESULTS, ROWS, SEEDS, VARIETIES, fold_split  # noqa: E402  (also sets sys.path)
from data import load_leaf_sizes  # noqa: E402
from logistic_regression import (BANDS, INK, MUTED, N_BOOT, SEED, choose_C, evaluate, get_device,  # noqa: E402
                                 get_logger, load_dataset, load_features, make_model, nested_row_cv,
                                 plot_confusion)
from logistic_regression import Config as LRConfig  # noqa: E402

FIG = HERE / "figures"
BAND_COLOR = {"all": "#2a78d6", "rgb": "#eb6834"}
GRID = "#e1e0d9"


def load_ensemble(run: str, n: int, key: str = "prob") -> tuple[np.ndarray, np.ndarray]:
    """Collect one run's test probabilities and average them over the seeds.

    n: the number of leaves. s: the number of seeds.

    :param run: Results folder name, e.g. all or rgb_no_val_e37.
    :type run: str
    :param n: Number of leaves.
    :type n: int
    :param key: Which saved probabilities: prob, or prob_cnn / prob_spectrum of a fusion run.
    :type key: str
    :return: Ensemble probabilities (n, 4) and per-seed probabilities (s, n, 4).
    :rtype: tuple
    """
    P = np.full((len(SEEDS), n, 4), np.nan)
    for s in SEEDS:
        for k in range(5):
            f = np.load(RESULTS / run / f"seed{s}" / f"fold{k}.npz")
            P[s, f["test_idx"]] = f[key]
    assert not np.isnan(P).any(), f"missing folds for {run}"
    return P.mean(0), P


def lr_three_rows(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Logistic regression on the same 3 training rows as the CNN, so neither has more data.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Variety codes, length n.
    :type y: np.ndarray
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :return: Out-of-fold probabilities, shape (n, 4).
    :rtype: np.ndarray
    """
    P = np.empty((len(y), 4))
    for k in range(5):
        tr, _va, te = fold_split(rows, k)
        best = choose_C(X[tr], y[tr], rows[tr])
        P[te] = make_model(best).fit(X[tr], y[tr]).predict_proba(X[te])
    return P


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
    P, _folds = nested_row_cv(sizes, y, meta.row.to_numpy().astype(str))
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

    :param run: Results folder name, e.g. all or rgb_no_val_e37.
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
    # The nested and fusion headline: the 5 row scores averaged, for each single model.
    m["seed_row_means"] = [float(np.mean([balanced_accuracy_score(y[rows == r], p[rows == r].argmax(1)) for r in ROWS]))
                           for p in seeds]
    m["median_best_epoch"] = float(np.median([np.load(RESULTS / run / f"seed{s}" / f"fold{k}.npz")["best_epoch"]
                                              for s in SEEDS for k in range(5)]))
    return m


def compute(device: torch.device) -> dict:
    """Score the CNN on each input, its logistic-regression partners and the size control.

    :param device: Where the bootstrap runs.
    :type device: torch.device
    :return: {"cnn": per configuration, "lr3": logistic regression on the CNN's 3 rows, "lr4": on 4 rows,
        "deltas": paired differences}.
    :rtype: dict
    """
    meta, _w, _m, _g = load_dataset()
    rows = meta.row.to_numpy().astype(str)
    out = {"cnn": {}, "lr3": {}, "lr4": {}, "deltas": {}, "chosen_epochs": {}, "fusion_parts": {}}
    for b in BANDS:
        d = load_features(LRConfig(b), device)
        y = d["y"]
        out["lr3"][b] = evaluate(y, lr_three_rows(d["X"], y, rows), device)
        out["lr4"][b] = evaluate(y, nested_row_cv(d["X"], y, rows)[0], device)
        out["cnn"][b] = score_run(b, y, rows, device)
        out["cnn"][f"{b}_no_val_e37"] = score_run(f"{b}_no_val_e37", y, rows, device)
        out["cnn"][f"{b}_nested"] = score_run(f"{b}_nested", y, rows, device)
        out["chosen_epochs"][b] = json.loads((RESULTS / f"{b}_nested" / "chosen_epochs.json").read_text())
        out["cnn"][f"{b}_fusion"] = score_run(f"{b}_fusion", y, rows, device)
        out["fusion_parts"][b] = {key: evaluate(y, load_ensemble(f"{b}_fusion", len(y), key)[0], device)
                                  for key in ("prob_cnn", "prob_spectrum")}
    cnn, lr3, lr4 = out["cnn"], out["lr3"], out["lr4"]
    out["deltas"] = {
        "CNN: 204 bands − RGB": paired_delta(y, cnn["all"]["pred"], cnn["rgb"]["pred"], device),
        "Logistic regression (3 rows): 204 bands − RGB": paired_delta(y, lr3["all"]["pred"], lr3["rgb"]["pred"], device),
        "204 bands: CNN − logistic regression (3 rows)": paired_delta(y, cnn["all"]["pred"], lr3["all"]["pred"], device),
        "RGB: CNN − logistic regression (3 rows)": paired_delta(y, cnn["rgb"]["pred"], lr3["rgb"]["pred"], device),
        "CNN without validation, 37 epochs each: 204 bands − RGB": paired_delta(
            y, cnn["all_no_val_e37"]["pred"], cnn["rgb_no_val_e37"]["pred"], device),
        "CNN, epochs chosen inside each split: 204 bands − RGB": paired_delta(
            y, cnn["all_nested"]["pred"], cnn["rgb_nested"]["pred"], device),
        "204 bands: CNN, epochs chosen inside each split − logistic regression (4 rows)": paired_delta(
            y, cnn["all_nested"]["pred"], lr4["all"]["pred"], device),
        "RGB: CNN, epochs chosen inside each split − logistic regression (4 rows)": paired_delta(
            y, cnn["rgb_nested"]["pred"], lr4["rgb"]["pred"], device),
        "Logistic regression (4 rows): 204 bands − RGB": paired_delta(y, lr4["all"]["pred"], lr4["rgb"]["pred"], device),
        "204 bands: CNN, 37 epochs, no validation − logistic regression (4 rows)": paired_delta(
            y, cnn["all_no_val_e37"]["pred"], lr4["all"]["pred"], device),
        "RGB: CNN, 37 epochs, no validation − logistic regression (4 rows)": paired_delta(
            y, cnn["rgb_no_val_e37"]["pred"], lr4["rgb"]["pred"], device),
        "CNN + mean spectrum: 204 bands − RGB": paired_delta(y, cnn["all_fusion"]["pred"], cnn["rgb_fusion"]["pred"], device),
    }
    for b, name in (("all", "204 bands"), ("rgb", "RGB")):
        fused = cnn[f"{b}_fusion"]["pred"]
        out["deltas"][f"{name}: CNN + mean spectrum − plain CNN"] = paired_delta(y, fused, cnn[b]["pred"], device)
        out["deltas"][f"{name}: CNN + mean spectrum − logistic regression (3 rows)"] = paired_delta(y, fused, lr3[b]["pred"], device)
        out["deltas"][f"{name}: CNN + mean spectrum − logistic regression (4 rows)"] = paired_delta(y, fused, lr4[b]["pred"], device)
    return out


def to_markdown(r: dict) -> str:
    """Build metrics.md.

    :param r: Output of :func:`compute`.
    :type r: dict
    :return: Markdown text.
    :rtype: str
    """
    def line(label: str, m: dict) -> str:
        """One table row: score with interval."""
        lo, hi = m["balanced_accuracy_ci95"]
        return f"| {label} | {m['balanced_accuracy']:.3f} ({lo:.3f}–{hi:.3f}) |"

    lines = ["# Variety CNN: metrics", "", "Balanced accuracy, chance 0.25; held-out field rows; 95% bootstrap interval. "
             "Every model uses SNV; CNN scores are 3-seed ensembles. `_no_val_e<N>` = trained on 4 rows for exactly N "
             "epochs, with no validation row. `_nested` = the same, with N chosen inside each training split. "
             "`_fusion` = the plain run, with the mean spectrum also in the last layer.", "",
             "| model | balanced accuracy |", "|---|---|"]
    lines += [line(f"CNN `{k}`", m) for k, m in r["cnn"].items()]
    lines += [line(f"logistic regression `{k}`, same 3 rows", m) for k, m in r["lr3"].items()]
    lines += [line(f"logistic regression `{k}`, 4 rows", m) for k, m in r["lr4"].items()]
    lines += ["", f"Leaf size alone (pixel count, height, width), logistic regression: {r['leaf_size']:.3f}.", "",
              "## Paired differences (same resampled leaves)", "", "| comparison | Δ balanced accuracy [95% CI] |", "|---|---|"]
    lines += [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in r["deltas"].items()]
    lines += ["", "## Epochs chosen inside each training split", "",
              "Average of the 5 row scores for each of the 3 single models, and the epoch count each test row got.", "",
              "| input | row average per seed | mean | " + " | ".join(ROWS) + " |", "|---|---|---|" + "---|" * len(ROWS)]
    for b, chosen in r["chosen_epochs"].items():
        means = r["cnn"][f"{b}_nested"]["seed_row_means"]
        lines.append(f"| `{b}` | {', '.join(f'{x:.3f}' for x in means)} | {np.mean(means):.3f} | "
                     + " | ".join(str(chosen[x]) for x in ROWS) + " |")
    lines += ["", "## CNN with the mean spectrum in its last layer", "",
              "Trained like the plain CNN (3 rows, early stopping). Row average per seed; each part alone is a 3-seed ensemble.", "",
              "| input | plain CNN, per seed | + mean spectrum, per seed | 48-d part alone | spectrum part alone |", "|---|---|---|---|---|"]
    for b, parts in r["fusion_parts"].items():
        plain, fused = r["cnn"][b]["seed_row_means"], r["cnn"][f"{b}_fusion"]["seed_row_means"]
        lines.append(f"| `{b}` | {', '.join(f'{x:.3f}' for x in plain)} | {', '.join(f'{x:.3f}' for x in fused)} | "
                     f"{parts['prob_cnn']['balanced_accuracy']:.3f} | {parts['prob_spectrum']['balanced_accuracy']:.3f} |")
    lines += ["", "## CNN detail", "", "| configuration | seeds | " + " | ".join(ROWS) + " | median best epoch |",
              "|---|---|" + "---|" * (len(ROWS) + 1)]
    lines += [f"| `{k}` | {', '.join(f'{s:.3f}' for s in m['seed_scores'])} | "
              + " | ".join(f"{m['per_row'][x]:.3f}" for x in ROWS) + f" | {m['median_best_epoch']:.0f} |"
              for k, m in r["cnn"].items()]
    return "\n".join(lines) + "\n"


def plot_scores(r: dict, path: Path) -> None:
    """Dot plot: each model's balanced accuracy with its interval, coloured by input bands.

    :param r: Output of :func:`compute`.
    :type r: dict
    :param path: Output file.
    :type path: Path
    """
    # Each row is (label, 204-band result, RGB result).
    groups = [("CNN + mean spectrum, 3 rows, early stopping", r["cnn"]["all_fusion"], r["cnn"]["rgb_fusion"]),
              ("CNN, 4 rows, epochs chosen in each split", r["cnn"]["all_nested"], r["cnn"]["rgb_nested"]),
              ("CNN, 4 rows, 37 epochs", r["cnn"]["all_no_val_e37"], r["cnn"]["rgb_no_val_e37"]),
              ("CNN, 3 rows, early stopping", r["cnn"]["all"], r["cnn"]["rgb"]),
              ("Logistic regression, same 3 rows", r["lr3"]["all"], r["lr3"]["rgb"]),
              ("Logistic regression, 4 rows", r["lr4"]["all"], r["lr4"]["rgb"])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
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


def write_outputs(r: dict) -> None:
    """Write metrics.md and the figures.

    :param r: Output of :func:`compute`.
    :type r: dict
    """
    FIG.mkdir(exist_ok=True)
    (RESULTS / "metrics.md").write_text(to_markdown(r))
    plot_scores(r, FIG / "scores.png")
    for name in ("all_no_val_e37", "rgb_no_val_e37"):
        m = r["cnn"][name]
        plot_confusion(m["confusion"], SimpleNamespace(name=name), m["balanced_accuracy"], FIG / f"confusion_{name}.png")


def main() -> None:
    """Load every run → metrics → write files."""
    log = get_logger("variety_cnn_report")
    device = get_device()
    write_outputs({**compute(device), "leaf_size": leaf_size_control(device)})
    log.info("[saved] results/metrics.md, figures/")


if __name__ == "__main__":
    main()
