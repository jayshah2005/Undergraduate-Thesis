"""Logistic regression on the leaf's mean spectrum: the disease baseline, and the leaves, labels and folds both models use."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.data import load_dataset  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from helpers.pca_common import VARIETIES, get_device, snv  # noqa: E402
from helpers.report_tables import BASELINE, METRICS_INTRO, cell, column_table, fold_note, main_table, table  # noqa: E402
from scoring import (METRICS, all_metrics, balanced_accuracy, bootstrap_weights,  # noqa: E402
                     per_variety_metrics, summary)

RES = HERE / "results"
ROWS = ["R1", "R2", "R3", "R4", "R5"]
DEFAULT_VARIETY = "Chardonnay"
# Pooled choices: one model over several varieties' leaves.
POOLS = {"all": list(VARIETIES), "baco_chardonnay_vidal": ["Baco", "Chardonnay", "Vidal"]}
VARIETY_CHOICES = [*VARIETIES, *POOLS]
# What counts as positive in each task; every other leaf is negative.
TASKS = {
    "leafroll": "leafroll present",
    "red_blotch": "red blotch present",
    "both": "both diseases present; a leaf with only one counts as negative",
    "any": "any disease present; healthy leaves are the negatives",
}
TASK_NAME = {"leafroll": "Leafroll", "red_blotch": "Red blotch", "both": "Both diseases",
             "any": "Any disease vs healthy"}  # section titles in reports
MIN_PER_CLASS = 5  # a variety with fewer positive or negative leaves sits out that task
BANDS = ("all", "rgb")  # full spectrum (204 bands), or only the 3 bands nearest RGB
RGB_NM = {"R": 640.0, "G": 550.0, "B": 460.0}  # the same picks as the variety models


# --------------------------------------------------------------------------
# Leaves, labels and folds (shared with ../cnn)
# --------------------------------------------------------------------------
def varieties_of(variety: str) -> list[str]:
    """Varieties covered by a --variety choice.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :return: Variety names.
    :rtype: list[str]
    """
    return POOLS.get(variety, [variety])


def is_pooled(variety: str) -> bool:
    """True when a --variety choice pools several varieties into one model."""
    return variety in POOLS


def variety_label(variety: str) -> str:
    """How titles name a --variety choice, e.g. "Chardonnay", "all varieties", "Baco + Chardonnay + Vidal"."""
    if variety == "all":
        return "all varieties"
    return " + ".join(varieties_of(variety))


def load_meta(variety: str):
    """Load the leaf table of a --variety choice, without the cubes.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :return: Leaf table (path, variety, row, vine, labels) with a fresh index.
    :rtype: pandas.DataFrame
    """
    meta = load_dataset()[0]
    return meta[meta.variety.isin(varieties_of(variety))].reset_index(drop=True)


def rgb_bands(wavelengths: np.ndarray) -> np.ndarray:
    """Find the band closest to each of the three RGB centres.

    :param wavelengths: Band centres in nm, shape (204,).
    :type wavelengths: np.ndarray
    :return: Band indices in R, G, B order, length 3.
    :rtype: np.ndarray
    """
    return np.array([int(np.abs(wavelengths - nm).argmin()) for nm in RGB_NM.values()])


def band_indices(bands: str) -> np.ndarray:
    """Indices of the bands a --bands choice keeps: all 204, or the 3 RGB bands.

    :param bands: A value of BANDS.
    :type bands: str
    :return: Band indices, length d (204 or 3).
    :rtype: np.ndarray
    """
    wavelengths = load_dataset()[1]
    return rgb_bands(wavelengths) if bands == "rgb" else np.arange(len(wavelengths))


def load_means(variety: str, device: torch.device, bands: str = "all") -> torch.Tensor:
    """Load the leaf-mean spectra of a --variety choice, in the order of :func:`load_meta`.

    n: leaves of the chosen varieties. d: bands kept (204 or 3).

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :param device: GPU or CPU.
    :type device: torch.device
    :param bands: A value of BANDS.
    :type bands: str
    :return: Leaf-mean reflectance, shape (n, d) float64.
    :rtype: torch.Tensor
    """
    meta, _w, means, _g = load_dataset()
    keep = meta.variety.isin(varieties_of(variety)).to_numpy()
    # Row-major like the unsliced array: a column-major copy makes SNV sum in another order, and the
    # pooled fit turns that 1e-15 change into visible score shifts.
    return torch.tensor(np.ascontiguousarray(means[keep][:, band_indices(bands)]), dtype=torch.float64, device=device)


def task_labels(meta, task: str) -> np.ndarray:
    """Give every leaf its 0/1 label for one task (see TASKS).

    :param meta: Leaf table with red_blotch and leafroll3, n rows.
    :type meta: pandas.DataFrame
    :param task: A key of TASKS.
    :type task: str
    :return: 0/1 label per leaf, length n.
    :rtype: np.ndarray
    """
    leafroll, red_blotch = meta.leafroll3.to_numpy() == 1, meta.red_blotch.to_numpy() == 1
    positive = {"leafroll": leafroll, "red_blotch": red_blotch,
                "both": leafroll & red_blotch, "any": leafroll | red_blotch}[task]
    return positive.astype(int)


def usable_leaves(meta, y: np.ndarray) -> np.ndarray:
    """Keep the varieties with at least MIN_PER_CLASS positive and negative leaves.

    Below that a variety can't be scored, and in a pooled model its leaves would only teach
    "this variety means this class". Cab Franc has 1 leaf with leafroll, for example.

    :param meta: Leaf table with variety, n rows.
    :type meta: pandas.DataFrame
    :param y: 0/1 label per leaf, length n.
    :type y: np.ndarray
    :return: True for the leaves this task uses, length n.
    :rtype: np.ndarray
    """
    varieties = meta.variety.to_numpy()
    keep = np.zeros(len(y), dtype=bool)
    for v in np.unique(varieties):
        in_variety = varieties == v
        if min(y[in_variety].sum(), (1 - y[in_variety]).sum()) >= MIN_PER_CLASS:
            keep |= in_variety
    return keep


def fold_of(meta) -> np.ndarray:
    """Give every leaf its fold 0-4: fold k holds field row k of every variety.

    :param meta: Leaf table with row, n rows.
    :type meta: pandas.DataFrame
    :return: Fold per leaf, length n.
    :rtype: np.ndarray
    """
    return np.array([ROWS.index(r) for r in meta.row.to_numpy()])


def row_groups(meta) -> np.ndarray:
    """Give every leaf its group: variety × row.

    The bootstrap resamples leaves within each group and class, so every resample keeps
    each row's size and disease rate. For one variety, a group is simply the row.

    :param meta: Leaf table with variety and row, n rows.
    :type meta: pandas.DataFrame
    :return: Group id per leaf, length n.
    :rtype: np.ndarray
    """
    varieties = sorted(set(meta.variety))
    variety_code = np.array([varieties.index(v) for v in meta.variety])
    return variety_code * len(ROWS) + fold_of(meta)


def fold_split(fold: np.ndarray, k: int, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = row k, validation = row k+1, train = the other 3 rows.

    The CNN stops early on the validation row. The baseline has no setting to choose, so it
    leaves that row unused and trains on the same 3 rows as the CNN.

    :param fold: Fold per leaf from :func:`fold_of`, length n.
    :type fold: np.ndarray
    :param k: Fold number 0-4.
    :type k: int
    :param keep: Leaves to split, from :func:`usable_leaves`; None means all of them.
    :type keep: np.ndarray | None
    :return: train, validation, test leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(fold)) if keep is None else np.flatnonzero(keep)
    f, val = fold[idx], (k + 1) % 5
    return idx[(f != k) & (f != val)], idx[f == val], idx[f == k]


def check_task(meta, fold: np.ndarray, task: str) -> None:
    """Stop before training if a task can't be run on these leaves and folds.

    :param meta: Leaf table with variety and labels, n rows.
    :type meta: pandas.DataFrame
    :param fold: Fold per leaf from :func:`fold_of`, length n.
    :type fold: np.ndarray
    :param task: A key of TASKS.
    :type task: str
    :raises ValueError: if no variety is usable, or a fold's train, validation or test set lacks a class.
    """
    y = task_labels(meta, task)
    keep = usable_leaves(meta, y)
    if not keep.any():
        raise ValueError(f"{task}: every variety here has fewer than {MIN_PER_CLASS} positive or negative leaves")
    for k in range(5):
        for name, part in zip(("train", "validation", "test"), fold_split(fold, k, keep)):
            if len(np.unique(y[part])) < 2:
                raise ValueError(f"{task}: the {name} set of test row {ROWS[k]} has only one class; "
                                 "try another variety")


# --------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------
def baseline_fold(means: torch.Tensor, y: np.ndarray, train_idx: np.ndarray,
                  test_idx: np.ndarray) -> np.ndarray:
    """Logistic regression on the SNV leaf-mean spectrum (the earlier baseline).

    Why scikit-learn: it is the model the archived per-variety baseline used
    (StandardScaler + LogisticRegression, class_weight="balanced"). SNV runs over the bands
    kept, so with RGB it works over 3 values only, as in the variety models.

    :param means: Leaf-mean reflectance, shape (n, d).
    :type means: torch.Tensor
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param train_idx: Training leaves.
    :type train_idx: np.ndarray
    :param test_idx: Test leaves.
    :type test_idx: np.ndarray
    :return: P(positive) for the test leaves, length n_test.
    :rtype: np.ndarray
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Z = snv(means.double()).cpu().numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(Z[train_idx], y[train_idx])
    return clf.predict_proba(Z[test_idx])[:, 1]


def run_baselines(meta, means: torch.Tensor, tasks: list[str]) -> dict:
    """Score the baseline on every row, fitting on the same 3 rows the CNN trains on.

    n: leaves of the chosen varieties.

    :param meta: Leaf table from :func:`load_meta`, n rows.
    :type meta: pandas.DataFrame
    :param means: Leaf-mean reflectance from :func:`load_means`, shape (n, d).
    :type means: torch.Tensor
    :param tasks: Keys of TASKS.
    :type tasks: list[str]
    :return: {task: P(positive) per leaf, length n; 0 for leaves the task leaves out}.
    :rtype: dict
    """
    fold = fold_of(meta)
    out = {}
    for task in tasks:
        y = task_labels(meta, task)
        keep = usable_leaves(meta, y)
        p = np.zeros(len(y))
        for k in range(5):
            tr, _va, te = fold_split(fold, k, keep)
            p[te] = baseline_fold(means, y, tr, te)
        out[task] = p
    return out


def runnable_tasks(meta) -> list[str]:
    """The tasks of TASKS that :func:`check_task` accepts for these leaves."""
    fold, ok = fold_of(meta), []
    for task in TASKS:
        try:
            check_task(meta, fold, task)
            ok.append(task)
        except ValueError:
            pass
    return ok


def score_baselines(meta, base: dict, device: torch.device) -> dict:
    """Balanced accuracy of the baseline for each task: overall with a 95% interval, per row and per variety.

    The interval uses the same seeded resamples as the CNN report, so the two reports give the
    baseline the same interval.

    :param meta: Leaf table from :func:`load_meta`.
    :type meta: pandas.DataFrame
    :param base: Output of :func:`run_baselines`.
    :type base: dict
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: output of :func:`scoring.summary` plus ci95, n_pos, n_neg, left_out, all
        (every metric, from :func:`scoring.all_metrics`) and all_by_variety}.
    :rtype: dict
    """
    fold, group, varieties = fold_of(meta), row_groups(meta), meta.variety.to_numpy()
    out = {}
    for task, p in base.items():
        y = task_labels(meta, task)
        keep = usable_leaves(meta, y)
        s = summary(p[keep][None], y[keep], fold[keep], varieties[keep], device)
        yt = torch.tensor(y[keep], dtype=torch.float64, device=device)
        gt = torch.tensor(group[keep], device=device)
        pt = torch.tensor(p[keep], dtype=torch.float64, device=device)
        w = bootstrap_weights(yt, gt)
        boot = balanced_accuracy(pt, yt, w)
        s["all"] = all_metrics(pt, yt, w)
        s["all_by_variety"] = per_variety_metrics(pt, yt, varieties[keep])
        s["ci95"] = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64, device=device)).tolist()
        s["n_pos"], s["n_neg"] = int(y[keep].sum()), int((1 - y[keep]).sum())
        s["left_out"] = sorted(set(varieties[~keep]))
        out[task] = s
    return out


def results_name(variety: str, bands: str = "all") -> str:
    """Folder name for a --variety and --bands choice, e.g. "chardonnay", "all" or "chardonnay_rgb"."""
    return variety.lower().replace(" ", "_") + ("_rgb" if bands == "rgb" else "")


def bands_label(bands: str) -> str:
    """How titles name a --bands choice."""
    return "RGB bands only" if bands == "rgb" else "full spectrum"


def final_report_markdown(variety: str, bands: str, scores: dict) -> str:
    """Build final_report.md in the shared format (report_tables): balanced accuracy, one section per task.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :param scores: Output of :func:`score_baselines`.
    :type scores: dict
    :return: Markdown text.
    :rtype: str
    """
    lines = [f"# Disease: logistic regression on {variety_label(variety)}, {bands_label(bands)}", "",
             "Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is "
             "used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% "
             "bootstrap interval. Every metric: `metrics.md`.", ""]
    for task, s in scores.items():
        lines += [f"## {TASK_NAME[task]} ({s['n_pos']} + / {s['n_neg']} −)", ""]
        lines += main_table([{"name": BASELINE, "bal_acc": s["bal_acc"], "ci95": s["ci95"],
                              "folds": list(s["by_fold"].values())}])
        lines += ["", fold_note(list(s["by_variety"]) or varieties_of(variety)), ""]
        if s["left_out"]:
            lines += [f"Left out (fewer than {MIN_PER_CLASS} positive or negative leaves): "
                      + ", ".join(s["left_out"]) + ".", ""]
        if s["by_variety"]:
            lines += column_table("Balanced accuracy per variety:", list(s["by_variety"]),
                                  {BASELINE: s["by_variety"]}) + [""]
    return "\n".join(lines)


METRIC_HEADER = {"bal_acc": "bal. acc.", "accuracy": "accuracy", "sensitivity": "sensitivity (recall)",
                 "specificity": "specificity", "precision": "precision", "f1": "F1"}


def metrics_section(models: dict) -> list[str]:
    """Every metric of one task: overall with intervals, confusion counts, then per variety for a pool.

    Shared with the CNN report, so both metrics.md files read the same way.

    :param models: {model name: {"all": output of scoring.all_metrics, "all_by_variety": per-variety values}}.
    :type models: dict
    :return: Markdown lines.
    :rtype: list[str]
    """
    lines = table(["model"] + [METRIC_HEADER[k] + " [95% CI]" for k in METRICS],
                  [[name] + [cell(m["all"][k], m["all"][k + "_ci95"]) for k in METRICS] for name, m in models.items()])
    lines += ["", "Confusion counts (positive = called diseased at P ≥ 0.5):", ""]
    lines += table(["model", "TP", "FN", "FP", "TN"],
                   [[name] + [str(round(m["all"][k])) for k in ("tp", "fn", "fp", "tn")] for name, m in models.items()])
    if any(m["all_by_variety"] for m in models.values()):
        lines += ["", "Per variety:", ""]
        lines += table(["model", "variety"] + [METRIC_HEADER[k] for k in METRICS],
                       [[name, v] + [cell(x[k]) for k in METRICS]
                        for name, m in models.items() for v, x in m["all_by_variety"].items()])
    return lines


def metrics_markdown(variety: str, bands: str, scores: dict) -> str:
    """Build metrics.md: every metric of the baseline, one section per task.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :param scores: Output of :func:`score_baselines`.
    :type scores: dict
    :return: Markdown text.
    :rtype: str
    """
    lines = [f"# Disease: logistic regression on {variety_label(variety)}, {bands_label(bands)}: all metrics", "",
             "A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. " + METRICS_INTRO, ""]
    for task, s in scores.items():
        lines += [f"## {TASK_NAME[task]} ({s['n_pos']} + / {s['n_neg']} −)", ""]
        lines += metrics_section({BASELINE: s}) + [""]
    return "\n".join(lines)


def main() -> None:
    """Parse --variety and --bands → load → baseline on every runnable task → metrics → write both reports."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", choices=VARIETY_CHOICES, default=DEFAULT_VARIETY,
                    help="one variety, or a pool (all, baco_chardonnay_vidal)")
    ap.add_argument("--bands", choices=BANDS, default="all",
                    help="all 204 bands, or only the 3 RGB bands; rgb adds _rgb to the folder name")
    args = ap.parse_args()
    log = get_logger("disease_logistic_regression")
    device = get_device()

    meta = load_meta(args.variety)
    base = run_baselines(meta, load_means(args.variety, device, args.bands), runnable_tasks(meta))
    scores = score_baselines(meta, base, device)

    res = RES / results_name(args.variety, args.bands)
    res.mkdir(parents=True, exist_ok=True)
    (res / "final_report.md").write_text(final_report_markdown(args.variety, args.bands, scores))
    (res / "metrics.md").write_text(metrics_markdown(args.variety, args.bands, scores))
    log.info("%s: %s | results/%s/{final_report,metrics}.md", args.variety,
             ", ".join(f"{t} {s['bal_acc']:.3f}" for t, s in scores.items()), res.name)


if __name__ == "__main__":
    main()
