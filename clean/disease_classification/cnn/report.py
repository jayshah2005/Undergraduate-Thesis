"""Score every arm against the mean-spectrum baseline and draw the report figures."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE.parent / "logistic_regression", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data import load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from logistic_regression import (ALL_VARIETIES, BANDS, DEFAULT_VARIETY, ROWS, TASKS, auc_groups,  # noqa: E402
                                 bands_label, fold_of, load_meta, load_means, rgb_bands, run_baselines,
                                 task_labels, usable_leaves)
from pca_common import VARIETIES, get_device  # noqa: E402
from run_experiments import SEEDS, load_cube_list, results_root, run_dir  # noqa: E402
from scoring import bootstrap_weights, within_row_auc  # noqa: E402
from cnn_common import pad_to_canvas  # noqa: E402
from train import BRIGHTNESS, POOLING, TrainConfig  # noqa: E402

FIG = HERE / "figures"
ARMS = [(b, p) for b in BRIGHTNESS for p in POOLING]
ARM_LABEL = {("band_z", "mean"): "band z-score + mean pool",
             ("band_z", "attention"): "band z-score + attention",
             ("snv", "mean"): "pixel SNV + mean pool",
             ("snv", "attention"): "pixel SNV + attention"}
TASK_NAME = {"leafroll": "Leafroll", "red_blotch": "Red blotch", "both": "Both diseases", "any": "Any disease vs healthy"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CNN_COLOR, BASE_COLOR = "#2a78d6", "#898781"
# Attention relative to uniform: blue below 1, gray at 1, red above.
REL_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rel", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])


# --------------------------------------------------------------------------
# Loading predictions
# --------------------------------------------------------------------------
def task_label(meta, task: str) -> str:
    """Title of one task, with its class counts over the leaves it uses.

    :param meta: Leaf table with variety and labels.
    :type meta: pandas.DataFrame
    :param task: A key of TASKS.
    :type task: str
    :return: e.g. "Leafroll (122 + / 128 −)".
    :rtype: str
    """
    y = task_labels(meta, task)
    y = y[usable_leaves(meta, y)]
    return f"{TASK_NAME[task]} ({y.sum()} + / {(1 - y).sum()} −)"


def tasks_in(root: Path) -> list[str]:
    """Tasks a results folder holds runs for, so reporting one task never drops another."""
    return [t for t in TASKS if (root / "runs" / t).exists()]


def load_predictions(meta, root: Path, variety: str, bands: str, device: torch.device) -> dict:
    """Collect out-of-fold probabilities for every task, arm and seed, plus the baselines.

    The baseline is refitted here (it takes seconds and is deterministic), as in the variety CNN report,
    on the same bands as the CNN.

    n: leaves the task uses. S4: side of the last feature map.

    :param meta: Leaf table from load_meta (variety, row, vine, labels).
    :type meta: pandas.DataFrame
    :param root: Results folder, from :func:`results_root`.
    :type root: Path
    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: {"y": (n,), "fold": (n,), "group": (n,), "variety": (n,), "leaf_idx": (n,) position in meta,
        "base": (n,), "cnn": {(brightness, pooling): (n_seeds, n)}, "att": {arm: (n, S4, S4)} (seed 0),
        "epochs": {arm: list of best epochs}}}.
    :rtype: dict
    """
    base = run_baselines(meta, load_means(variety, device, bands), tasks_in(root))
    fold = fold_of(meta)
    group = auc_groups(meta)
    variety = meta.variety.to_numpy()   # from here on: variety per leaf
    out = {}
    for task in tasks_in(root):
        y = task_labels(meta, task)
        keep = usable_leaves(meta, y)
        t = {"y": y[keep], "fold": fold[keep], "group": group[keep], "variety": variety[keep],
             "leaf_idx": np.flatnonzero(keep), "base": base[task][keep], "cnn": {}, "att": {}, "epochs": {}}
        for arm in ARMS:
            probs, epochs = np.full((len(SEEDS), len(meta)), np.nan), []
            for s in SEEDS:
                for k in range(5):
                    f = np.load(run_dir(task, *arm, s, root) / f"fold{k}.npz")
                    probs[s, f["test_idx"]] = f["prob"]
                    epochs.append(int(f["best_epoch"]))
                    if s == 0:
                        if arm not in t["att"]:
                            t["att"][arm] = np.zeros((len(meta),) + f["attention"].shape[1:], np.float32)
                        t["att"][arm][f["test_idx"]] = f["attention"]
            assert not np.isnan(probs[:, keep]).any(), f"missing folds for {task} {arm}"
            t["cnn"][arm], t["epochs"][arm] = probs[:, keep], epochs
            t["att"][arm] = t["att"][arm][keep]
        out[task] = t
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(preds: dict, device: torch.device) -> dict:
    """Score every arm and the baseline for each task.

    Seed-ensemble = the mean probability over the 3 seeds. Its 95% interval and
    the paired difference to the baseline come from the same 2000 resamples.

    :param preds: Output of :func:`load_predictions`.
    :type preds: dict
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: {name: {metric: value}}} with names "baseline" and the arm labels;
        plus per-row AUCs, and per-variety AUCs when a task uses several varieties.
    :rtype: dict
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # standard metric code

    out = {}
    for task, t in preds.items():
        y = torch.tensor(t["y"], dtype=torch.float64, device=device)
        group = torch.tensor(t["group"], device=device)     # AUC groups: variety × row
        fk = t["fold"]
        varieties = sorted(set(t["variety"]))
        w = bootstrap_weights(y, group)
        tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
        base_boot = within_row_auc(tt(t["base"]), y, group, w)

        def score(p_seeds: np.ndarray) -> dict:
            ens = p_seeds.mean(0)
            boot = within_row_auc(tt(ens), y, group, w)
            diff = boot - base_boot
            per_seed = [within_row_auc(tt(p), y, group).item() for p in p_seeds]
            per_fold = [within_row_auc(tt(ens)[tt(fk) == k], y[tt(fk) == k], group[tt(fk) == k]).item()
                        for k in range(5)]
            by_variety = {}
            for v in varieties if len(varieties) > 1 else []:
                in_v = torch.tensor(t["variety"] == v, device=device)
                by_variety[v] = within_row_auc(tt(ens)[in_v], y[in_v], group[in_v]).item()
            return {
                **({"per_variety_auc": by_variety} if by_variety else {}),
                "auc_seed_mean": float(np.mean(per_seed)), "auc_seed_sd": float(np.std(per_seed, ddof=1))
                if len(per_seed) > 1 else 0.0, "auc_seeds": per_seed,
                "auc_ens": within_row_auc(tt(ens), y, group).item(),
                "auc_ens_lo": torch.quantile(boot, 0.025).item(),
                "auc_ens_hi": torch.quantile(boot, 0.975).item(),
                "pooled_auc_ens": float(roc_auc_score(t["y"], ens)),
                "bal_acc_ens": float(balanced_accuracy_score(t["y"], ens >= 0.5)),
                "diff_vs_base": (within_row_auc(tt(ens), y, group) - within_row_auc(tt(t["base"]), y, group)).item(),
                "diff_lo": torch.quantile(diff, 0.025).item(),
                "diff_hi": torch.quantile(diff, 0.975).item(),
                "p_better": (diff > 0).double().mean().item(),
                "per_fold_auc": per_fold,
            }

        res = {"baseline": score(t["base"][None])}
        for arm in ARMS:
            res[ARM_LABEL[arm]] = score(t["cnn"][arm])
            res[ARM_LABEL[arm]]["median_best_epoch"] = float(np.median(t["epochs"][arm]))
        out[task] = res
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def to_markdown(metrics: dict, labels: dict, variety: str, bands: str) -> str:
    """Write the metric tables as Markdown.

    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param labels: {task: title from :func:`task_label`}.
    :type labels: dict
    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :return: Markdown text.
    :rtype: str
    """
    pooled = variety == ALL_VARIETIES
    title = "all varieties" if pooled else variety
    same = "variety and row" if pooled else "row"
    note = " Bal. acc. isn't compared within a variety, so recognising a variety can raise it." if pooled else ""
    lines = [f"# Whole-leaf CNN on {title}, {bands_label(bands)}: metrics", "",
             f"Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. "
             f"AUC compares leaves of the same {same} only (0.5 = chance). Δ: paired difference to the baseline.{note}", ""]
    for task, res in metrics.items():
        lines += [f"## {labels[task]}", "",
                  "| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |",
                  "|---|---|---|---|---|"]
        for name, m in res.items():
            seeds = f"{m['auc_seed_mean']:.3f} ± {m['auc_seed_sd']:.3f}" if "baseline" not in name else "—"
            delta = "—" if name == "baseline" else \
                f"{m['diff_vs_base']:+.3f} [{m['diff_lo']:+.3f}, {m['diff_hi']:+.3f}]"
            lines.append(f"| {name} | {seeds} | {m['auc_ens']:.3f} [{m['auc_ens_lo']:.3f}, {m['auc_ens_hi']:.3f}] "
                         f"| {delta} | {m['bal_acc_ens']:.3f} |")
        lines += ["", "AUC per test row (ensemble):", "",
                  "| model | " + " | ".join(ROWS) + " |", "|---|" + "---|" * 5]
        for name, m in res.items():
            lines.append(f"| {name} | " + " | ".join(f"{a:.3f}" for a in m["per_fold_auc"]) + " |")
        lines.append("")
        by_variety = next(iter(res.values())).get("per_variety_auc", {})
        if by_variety:
            lines += ["AUC per variety (ensemble):", "", "| model | " + " | ".join(by_variety) + " |",
                      "|---|" + "---|" * len(by_variety)]
            for name, m in res.items():
                lines.append(f"| {name} | " + " | ".join(f"{a:.3f}" for a in m["per_variety_auc"].values()) + " |")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_auc(metrics: dict, labels: dict, path: Path) -> None:
    """Dot plot: ensemble within-row AUC with 95% CI, seeds as small dots, per task.

    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param labels: {task: title from :func:`task_label`}.
    :type labels: dict
    :param path: Output PNG.
    :type path: Path
    """
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    names = ["baseline"] + [ARM_LABEL[a] for a in ARMS]
    # One task keeps the width of two panels, so its title and the model names still fit.
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * max(len(metrics), 2), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (task, res) in zip(axes, metrics.items()):
        for i, name in enumerate(names):
            m = res[name]
            y = len(names) - 1 - i
            color = BASE_COLOR if "baseline" in name else CNN_COLOR
            ax.plot([m["auc_ens_lo"], m["auc_ens_hi"]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(m["auc_ens"], y, "o", color=color, ms=8, mec="white", mew=1.5, zorder=3)
            if "baseline" not in name:
                ax.plot(m["auc_seeds"], [y - 0.22] * len(m["auc_seeds"]), "|", color=color, ms=7, alpha=0.8)
        ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_title(labels[task], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("within-row ROC AUC", fontsize=10, color=INK_2)
        _set_auc_limits(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names[::-1])
    fig.text(0.99, 0.01, "dot = 3-seed ensemble with 95% bootstrap CI; ticks = single seeds; dashed = chance",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _set_auc_limits(ax) -> None:
    """Use the usual AUC axis, widened if a point or interval would fall outside it."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(0.3, lo), max(0.85, hi))


def false_colour(X: torch.Tensor, mask: torch.Tensor, w: np.ndarray) -> np.ndarray:
    """RGB pictures of leaves from the bands nearest 640, 550 and 460 nm.

    k: leaves drawn. S: canvas side.

    :param X: Reflectance canvas, shape (k, 204, S, S).
    :type X: torch.Tensor
    :param mask: Leaf mask, shape (k, 1, S, S).
    :type mask: torch.Tensor
    :param w: Band centres in nm, shape (204,).
    :type w: np.ndarray
    :return: RGB in [0, 1], white background, shape (k, S, S, 3).
    :rtype: np.ndarray
    """
    rgb = X[:, rgb_bands(w)].float()
    leaf = mask.bool().expand_as(rgb)
    rgb = (rgb / torch.quantile(rgb[leaf][:200000], 0.99)).clamp(0, 1)
    rgb = torch.where(leaf, rgb, torch.ones_like(rgb))
    return rgb.permute(0, 2, 3, 1).cpu().numpy()


def fig_attention(preds: dict, metrics: dict, meta, variety: str, path: Path, device: torch.device) -> dict:
    """Attention maps of the best attention arm per task, on false-colour leaves (seed 0).

    Shows the 4 positive test leaves the model scored highest and the 2 negatives it
    scored lowest. Each map cell covers 4 × 4 pixels; its receptive field is ~26 px.

    :param preds: Output of :func:`load_predictions`.
    :type preds: dict
    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param meta: Leaf table from load_meta.
    :type meta: pandas.DataFrame
    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :param path: Output PNG.
    :type path: Path
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: (arm label, leaf paths shown)}.
    :rtype: dict
    """
    X, mask, side = pad_to_canvas(load_cube_list(variety), device)
    _m, w, _x, _g = load_dataset()
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    fig, axes = plt.subplots(len(preds), 6, figsize=(13, 2.5 * len(preds)), squeeze=False)
    shown = {}
    for r, (task, t) in enumerate(preds.items()):
        att_arms = [a for a in ARMS if a[1] == "attention"]
        arm = max(att_arms, key=lambda a: metrics[task][ARM_LABEL[a]]["auc_ens"])
        p = t["cnn"][arm][0]
        pos = np.flatnonzero(t["y"] == 1)
        neg = np.flatnonzero(t["y"] == 0)
        pick = list(pos[np.argsort(-p[pos])][:4]) + list(neg[np.argsort(p[neg])][:2])
        leaves = t["leaf_idx"][pick]    # the task may leave a variety out; X and meta hold every leaf
        rgb = false_colour(X[leaves], mask[leaves], w)
        att = torch.tensor(t["att"][arm][pick], device=device)[:, None]
        # Weight × number of valid positions: 1 = the weight mean pooling would give.
        # A fixed scale (not each map's own max) shows how flat the attention really is.
        att = att * (att > 0).flatten(1).sum(1)[:, None, None, None]
        att = F.interpolate(att, size=(side, side), mode="nearest")[:, 0].cpu().numpy()
        for c, (i, ax) in enumerate(zip(pick, axes[r])):
            leaf = mask[leaves[c], 0].cpu().numpy() > 0
            a = np.where(leaf & (att[c] > 0), att[c], np.nan)
            ax.imshow(rgb[c])
            im = ax.imshow(a, cmap=REL_MAP, alpha=0.75, vmin=0.0, vmax=2.0)
            ys, xs = np.nonzero(leaf)
            pad = 4
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.max() + pad, ys.min() - pad)
            ax.axis("off")
            lab = "+" if t["y"][i] == 1 else "−"
            name = meta.path[leaves[c]] if variety == ALL_VARIETIES else meta.path[leaves[c]].split("/", 1)[1]
            ax.set_title(f"{name}  ({lab})\nP = {p[i]:.2f}", fontsize=8.5, color=INK_2)
        axes[r, 0].text(-0.15, 0.5, f"{task.replace('_', ' ')}\n{ARM_LABEL[arm]}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9.5, color=INK, rotation=90)
        shown[task] = (ARM_LABEL[arm], [meta.path[i] for i in leaves])
    fig.text(0.99, 0.01, "attention weight relative to uniform (1 = same as mean pooling; uncoloured rim = "
             "excluded margin), seed 0 test predictions; left 4 = highest-scored positives, "
             "right 2 = lowest-scored negatives", ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.93, 1))
    cax = fig.add_axes([0.94, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5, 2]).ax.tick_params(labelsize=8, colors=INK_2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return shown


def main() -> None:
    """Load predictions → metrics → tables and figures."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", choices=[*VARIETIES, ALL_VARIETIES], default=DEFAULT_VARIETY,
                    help="report the runs from run_experiments.py --variety <value>")
    ap.add_argument("--bands", choices=BANDS, default="all",
                    help="report the runs from run_experiments.py --bands <value>")
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout,
                    help="report the runs from run_experiments.py --dropout <value>")
    ap.add_argument("--no-augmentation", action="store_true",
                    help="report the runs from run_experiments.py --no-augmentation")
    args = ap.parse_args()
    log = get_logger("disease_cnn_report")

    meta = load_meta(args.variety)
    root = results_root(False, args.dropout, args.variety, not args.no_augmentation, args.bands)
    labels = {t: task_label(meta, t) for t in tasks_in(root)}
    fig_dir = FIG / root.name
    device = get_device()
    preds = load_predictions(meta, root, args.variety, args.bands, device)
    metrics = compute_metrics(preds, device)

    (root / "metrics.md").write_text(to_markdown(metrics, labels, args.variety, args.bands))
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_auc(metrics, labels, fig_dir / "auc_comparison.png")
    fig_attention(preds, metrics, meta, args.variety, fig_dir / "attention_maps.png", device)
    log.info("[saved] results/%s/metrics.md", root.name)


if __name__ == "__main__":
    main()
