"""Score every arm against the mean-spectrum baseline and draw the report figures."""
from __future__ import annotations

import json
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
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data import load_cubes, load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from pca_common import get_device  # noqa: E402
from run_experiments import RESULTS, SEEDS, VARIETY, results_root, run_dir, variant_suffix  # noqa: E402
from train import (BRIGHTNESS, FOLD_SCHEMES, POOLING, ROWS, TASKS, TrainConfig, fold_of,  # noqa: E402
                   pad_to_canvas)

FIG = HERE / "figures"
N_BOOT, BOOT_SEED = 2000, 0
ARMS = [(b, p) for b in BRIGHTNESS for p in POOLING]
ARM_LABEL = {("band_z", "mean"): "band z-score + mean pool",
             ("band_z", "attention"): "band z-score + attention",
             ("snv", "mean"): "pixel SNV + mean pool",
             ("snv", "attention"): "pixel SNV + attention"}
TASK_LABEL = {"leafroll": "Leafroll (122 + / 128 −)", "red_blotch": "Red blotch (88 + / 162 −)"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CNN_COLOR, BASE_COLOR = "#2a78d6", "#898781"
# Attention relative to uniform: blue below 1, gray at 1, red above.
REL_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rel", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])


# --------------------------------------------------------------------------
# Loading predictions
# --------------------------------------------------------------------------
def load_predictions(meta, scheme: str, root: Path) -> dict:
    """Collect out-of-fold probabilities for every arm and seed, plus the baselines.

    group = fold × row: AUCs only compare leaves of one group, i.e. leaves scored by
    the same model and from the same row (so a row's disease rate cannot help).
    Under the row scheme a group is simply the test row.

    n: leaves (250). S4: side of the last feature map.

    :param meta: Leaf table from load_cubes (row, vine, labels).
    :type meta: pandas.DataFrame
    :param scheme: Fold scheme, "row", "photo" or "block".
    :type scheme: str
    :param root: Results folder of that scheme, from :func:`results_root`.
    :type root: Path
    :return: {task: {"y": (n,), "fold": (n,), "group": (n,), "base4" / "base3": (n,),
        "cnn": {(brightness, pooling): (n_seeds, n)}, "att": {arm: (n, S4, S4)} (seed 0),
        "epochs": {arm: list of best epochs}}}.
    :rtype: dict
    """
    fold = fold_of(meta, scheme)
    group = fold * 5 + np.array([ROWS.index(r) for r in meta.row])
    base = np.load(root / "baseline.npz")
    out = {}
    for task, col in TASKS.items():
        t = {"y": meta[col].to_numpy(), "fold": fold, "group": group, "base4": base[f"{task}__4rows"],
             "base3": base[f"{task}__3rows"], "cnn": {}, "att": {}, "epochs": {}}
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
            assert not np.isnan(probs).any(), f"missing folds for {task} {arm}"
            t["cnn"][arm], t["epochs"][arm] = probs, epochs
        out[task] = t
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _pair_tables(p: torch.Tensor, y: torch.Tensor, group: torch.Tensor) -> list:
    """Per group: 1 / 0.5 / 0 for each (positive, negative) pair, by who scores higher."""
    tabs = []
    for k in torch.unique(group).tolist():
        pk, yk = p[group == k], y[group == k]
        pos, neg = pk[yk == 1], pk[yk == 0]
        tabs.append((k, (pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()))
    return tabs


def within_fold_auc(p: torch.Tensor, y: torch.Tensor, group: torch.Tensor,
                    w: torch.Tensor | None = None) -> torch.Tensor:
    """ROC AUC that only compares leaves of the same group (same fold model, same row).

    This is the archived study's "within-variety AUC" restricted to one variety:
    a leaf is compared only with leaves scored by the same fold model, so
    differences between the five models (e.g. a shifted threshold) cannot help or
    hurt; and only with leaves of its own row, so a row's disease rate cannot help.
    Groups with no positive or no negative leaf contribute no pairs.
    With bootstrap weights w (B, n), returns one AUC per bootstrap sample.

    :param p: Probabilities, shape (n,).
    :type p: torch.Tensor
    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param group: Group id per leaf (fold × row), shape (n,).
    :type group: torch.Tensor
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: AUC, shape () or (B,).
    :rtype: torch.Tensor
    """
    num = den = 0.0
    for k, tab in _pair_tables(p, y, group):
        mk = group == k
        if w is None:
            num, den = num + tab.sum(), den + tab.numel()
        else:
            wk, yk = w[:, mk], y[mk]
            wp, wn = wk[:, yk == 1], wk[:, yk == 0]
            num = num + ((wp @ tab) * wn).sum(1)
            den = den + wp.sum(1) * wn.sum(1)
    return num / den


def bootstrap_weights(y: torch.Tensor, group: torch.Tensor, n_boot: int, seed: int) -> torch.Tensor:
    """Resample leaves with replacement within each (group, class) cell.

    Keeping each cell's size fixed keeps every bootstrap AUC defined and leaves
    the group structure intact. Seeded so every arm sees the same resamples, which
    makes the arm-minus-baseline differences paired.

    :param y: 0/1 labels, shape (n,).
    :type y: torch.Tensor
    :param group: Group per leaf (fold × row), shape (n,).
    :type group: torch.Tensor
    :param n_boot: Number of resamples B.
    :type n_boot: int
    :param seed: Generator seed.
    :type seed: int
    :return: Counts, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    g = torch.Generator(device=y.device).manual_seed(seed)
    w = torch.zeros(n_boot, len(y), dtype=torch.float64, device=y.device)
    for k in torch.unique(group).tolist():
        for c in (0, 1):
            idx = torch.nonzero((group == k) & (y == c)).squeeze(1)
            if len(idx) == 0:
                continue
            draw = idx[torch.randint(len(idx), (n_boot, len(idx)), generator=g, device=y.device)]
            w.scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
    return w


def compute_metrics(preds: dict, device: torch.device) -> dict:
    """Score every arm and both baselines for each task.

    Seed-ensemble = the mean probability over the 3 seeds. Its 95% interval and
    the paired difference to the baseline come from the same 2000 resamples.

    :param preds: Output of :func:`load_predictions`.
    :type preds: dict
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: {name: {metric: value}}} with names "baseline (4 folds)",
        "baseline (3 folds)" and the arm labels; plus per-fold AUCs.
    :rtype: dict
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # standard metric code

    out = {}
    for task, t in preds.items():
        y = torch.tensor(t["y"], dtype=torch.float64, device=device)
        fold = torch.tensor(t["group"], device=device)     # AUC groups: fold × row
        fk = t["fold"]
        w = bootstrap_weights(y, fold, N_BOOT, BOOT_SEED)
        tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
        base_boot = within_fold_auc(tt(t["base4"]), y, fold, w)

        def score(p_seeds: np.ndarray) -> dict:
            ens = p_seeds.mean(0)
            boot = within_fold_auc(tt(ens), y, fold, w)
            diff = boot - base_boot
            per_seed = [within_fold_auc(tt(p), y, fold).item() for p in p_seeds]
            per_fold = [within_fold_auc(tt(ens)[tt(fk) == k], y[tt(fk) == k], fold[tt(fk) == k]).item()
                        for k in range(5)]
            return {
                "auc_seed_mean": float(np.mean(per_seed)), "auc_seed_sd": float(np.std(per_seed, ddof=1))
                if len(per_seed) > 1 else 0.0, "auc_seeds": per_seed,
                "auc_ens": within_fold_auc(tt(ens), y, fold).item(),
                "auc_ens_lo": torch.quantile(boot, 0.025).item(),
                "auc_ens_hi": torch.quantile(boot, 0.975).item(),
                "pooled_auc_ens": float(roc_auc_score(t["y"], ens)),
                "bal_acc_ens": float(balanced_accuracy_score(t["y"], ens >= 0.5)),
                "diff_vs_base": (within_fold_auc(tt(ens), y, fold) - within_fold_auc(tt(t["base4"]), y, fold)).item(),
                "diff_lo": torch.quantile(diff, 0.025).item(),
                "diff_hi": torch.quantile(diff, 0.975).item(),
                "p_better": (diff > 0).double().mean().item(),
                "per_fold_auc": per_fold,
            }

        res = {"baseline (4 folds)": score(t["base4"][None]), "baseline (3 folds)": score(t["base3"][None])}
        for arm in ARMS:
            res[ARM_LABEL[arm]] = score(t["cnn"][arm])
            res[ARM_LABEL[arm]]["median_best_epoch"] = float(np.median(t["epochs"][arm]))
        out[task] = res
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def to_markdown(metrics: dict, scheme: str) -> str:
    """Write the metric tables as Markdown.

    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param scheme: Fold scheme the metrics come from.
    :type scheme: str
    :return: Markdown text.
    :rtype: str
    """
    fold_names = ROWS if scheme == "row" else [f"fold {k}" for k in range(5)]
    lines = [f"# Whole-leaf CNN on {VARIETY}: metrics, `{scheme}` folds", "",
             "AUC compares leaves of the same fold and row only (0.5 = chance). Δ: paired difference to the "
             "4-fold baseline.", ""]
    for task, res in metrics.items():
        lines += [f"## {TASK_LABEL[task]}", "",
                  "| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |",
                  "|---|---|---|---|---|"]
        for name, m in res.items():
            seeds = f"{m['auc_seed_mean']:.3f} ± {m['auc_seed_sd']:.3f}" if "baseline" not in name else "—"
            delta = "—" if name == "baseline (4 folds)" else \
                f"{m['diff_vs_base']:+.3f} [{m['diff_lo']:+.3f}, {m['diff_hi']:+.3f}]"
            lines.append(f"| {name} | {seeds} | {m['auc_ens']:.3f} [{m['auc_ens_lo']:.3f}, {m['auc_ens_hi']:.3f}] "
                         f"| {delta} | {m['bal_acc_ens']:.3f} |")
        lines += ["", "AUC per test fold (ensemble):", "",
                  "| model | " + " | ".join(fold_names) + " |", "|---|" + "---|" * 5]
        for name, m in res.items():
            lines.append(f"| {name} | " + " | ".join(f"{a:.3f}" for a in m["per_fold_auc"]) + " |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_auc(metrics: dict, path: Path) -> None:
    """Dot plot: ensemble within-fold AUC with 95% CI, seeds as small dots, per task.

    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param path: Output PNG.
    :type path: Path
    """
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    names = ["baseline (4 folds)", "baseline (3 folds)"] + [ARM_LABEL[a] for a in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
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
        ax.set_title(TASK_LABEL[task], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("within-fold ROC AUC", fontsize=10, color=INK_2)
        ax.set_xlim(0.3, 0.85)
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
    bands = [int(np.abs(w - nm).argmin()) for nm in (640.0, 550.0, 460.0)]
    rgb = X[:, bands].float()
    leaf = mask.bool().expand_as(rgb)
    rgb = (rgb / torch.quantile(rgb[leaf][:200000], 0.99)).clamp(0, 1)
    rgb = torch.where(leaf, rgb, torch.ones_like(rgb))
    return rgb.permute(0, 2, 3, 1).cpu().numpy()


def fig_attention(preds: dict, metrics: dict, path: Path, device: torch.device) -> dict:
    """Attention maps of the best attention arm per task, on false-colour leaves (seed 0).

    Shows the 4 positive test leaves the model scored highest and the 2 negatives it
    scored lowest. Each map cell covers 4 × 4 pixels; its receptive field is ~26 px.

    :param preds: Output of :func:`load_predictions`.
    :type preds: dict
    :param metrics: Output of :func:`compute_metrics`.
    :type metrics: dict
    :param path: Output PNG.
    :type path: Path
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {task: (arm label, leaf paths shown)}.
    :rtype: dict
    """
    meta, cubes = load_cubes(VARIETY)
    X, mask, side = pad_to_canvas(cubes, device)
    _m, w, _x, _g = load_dataset()
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 6, figsize=(13, 5.0))
    shown = {}
    for r, (task, t) in enumerate(preds.items()):
        att_arms = [a for a in ARMS if a[1] == "attention"]
        arm = max(att_arms, key=lambda a: metrics[task][ARM_LABEL[a]]["auc_ens"])
        p = t["cnn"][arm][0]
        pos = np.flatnonzero(t["y"] == 1)
        neg = np.flatnonzero(t["y"] == 0)
        pick = list(pos[np.argsort(-p[pos])][:4]) + list(neg[np.argsort(p[neg])][:2])
        rgb = false_colour(X[pick], mask[pick], w)
        att = torch.tensor(t["att"][arm][pick], device=device)[:, None]
        # Weight × number of valid positions: 1 = the weight mean pooling would give.
        # A fixed scale (not each map's own max) shows how flat the attention really is.
        att = att * (att > 0).flatten(1).sum(1)[:, None, None, None]
        att = F.interpolate(att, size=(side, side), mode="nearest")[:, 0].cpu().numpy()
        for c, (i, ax) in enumerate(zip(pick, axes[r])):
            leaf = mask[i, 0].cpu().numpy() > 0
            a = np.where(leaf & (att[c] > 0), att[c], np.nan)
            ax.imshow(rgb[c])
            im = ax.imshow(a, cmap=REL_MAP, alpha=0.75, vmin=0.0, vmax=2.0)
            ys, xs = np.nonzero(leaf)
            pad = 4
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.max() + pad, ys.min() - pad)
            ax.axis("off")
            lab = "+" if t["y"][i] == 1 else "−"
            ax.set_title(f"{meta.path[i].split('/', 1)[1]}  ({lab})\nP = {p[i]:.2f}", fontsize=8.5, color=INK_2)
        axes[r, 0].text(-0.15, 0.5, f"{task.replace('_', ' ')}\n{ARM_LABEL[arm]}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9.5, color=INK, rotation=90)
        shown[task] = (ARM_LABEL[arm], [meta.path[i] for i in pick])
    fig.text(0.99, 0.01, "attention weight relative to uniform (1 = same as mean pooling; uncoloured rim = "
             "excluded margin), seed 0 test predictions; left 4 = highest-scored positives, "
             "right 2 = lowest-scored negatives", ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.93, 1))
    cax = fig.add_axes([0.94, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5, 2]).ax.tick_params(labelsize=8, colors=INK_2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return shown


def comparison_markdown(all_metrics: dict) -> str:
    """One table per task: each model's ensemble AUC under every fold scheme.

    :param all_metrics: {scheme: output of :func:`compute_metrics`}.
    :type all_metrics: dict
    :return: Markdown text.
    :rtype: str
    """
    schemes = list(all_metrics)
    lines = [f"# Fold schemes compared ({VARIETY})", ""]
    lines += [f"- `{s}`: {FOLD_SCHEMES[s]}" for s in schemes]
    lines += ["", "Within-fold AUC (same fold and same row only), 3-seed ensemble [95% bootstrap CI]; "
              "Δ = paired difference to the 4-fold baseline of the same scheme.", ""]
    for task in TASKS:
        lines += [f"## {TASK_LABEL[task]}", "", "| model | " + " | ".join(f"`{s}`" for s in schemes) + " |",
                  "|---|" + "---|" * len(schemes)]
        for name in all_metrics[schemes[0]][task]:
            cells = []
            for s in schemes:
                m = all_metrics[s][task][name]
                cell = f"{m['auc_ens']:.3f} [{m['auc_ens_lo']:.3f}, {m['auc_ens_hi']:.3f}]"
                if name != "baseline (4 folds)":
                    cell += f"<br>Δ {m['diff_vs_base']:+.3f} [{m['diff_lo']:+.3f}, {m['diff_hi']:+.3f}]"
                cells.append(cell)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def fig_comparison(all_metrics: dict, path: Path) -> None:
    """Dot plot of every model's ensemble AUC with 95% CI, one marker per fold scheme.

    :param all_metrics: {scheme: output of :func:`compute_metrics`}.
    :type all_metrics: dict
    :param path: Output PNG.
    :type path: Path
    """
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    schemes = list(all_metrics)
    colors = dict(zip(schemes, ["#2a78d6", "#eb6834", "#1baf7a"]))
    names = list(all_metrics[schemes[0]]["leafroll"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        for i, name in enumerate(names):
            for j, s in enumerate(schemes):
                m = all_metrics[s][task][name]
                y = len(names) - 1 - i + (1 - j) * 0.22
                ax.plot([m["auc_ens_lo"], m["auc_ens_hi"]], [y, y], color=colors[s], lw=1.6,
                        solid_capstyle="round")
                ax.plot(m["auc_ens"], y, "o", color=colors[s], ms=6, mec="white", mew=1.2, zorder=3,
                        label=s if i == 0 else None)
        ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_title(TASK_LABEL[task], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("within-fold ROC AUC (same fold and row)", fontsize=10, color=INK_2)
        ax.set_xlim(0.3, 0.85)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names[::-1])
    axes[1].legend(title="folds", fontsize=9, title_fontsize=9, frameon=False, loc="lower right")
    fig.text(0.99, 0.01, "dot = 3-seed ensemble with 95% bootstrap CI; dashed = chance",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Load predictions → metrics → tables and figures, for one fold scheme or a comparison."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", choices=list(FOLD_SCHEMES), default="row")
    ap.add_argument("--compare", action="store_true",
                    help="compare every scheme that has a metrics.json (run each scheme's report first)")
    ap.add_argument("--no-brightness-jitter", action="store_true",
                    help="report the runs from run_experiments.py --no-brightness-jitter")
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout,
                    help="report the runs from run_experiments.py --dropout <value>")
    args = ap.parse_args()
    log = get_logger("attention_mil_report")

    jitter = not args.no_brightness_jitter
    if args.compare:
        roots = {s: results_root(s, False, jitter, args.dropout) for s in FOLD_SCHEMES}
        all_metrics = {s: json.loads((r / "metrics.json").read_text())
                       for s, r in roots.items() if (r / "metrics.json").exists()}
        name = "fold_comparison" + variant_suffix(jitter, args.dropout)
        FIG.mkdir(exist_ok=True)
        (RESULTS / f"{name}.md").write_text(comparison_markdown(all_metrics))
        fig_comparison(all_metrics, FIG / f"{name}.png")
        log.info("[saved] results/%s.md, figures/%s.png", name, name)
        return

    root = results_root(args.folds, False, jitter, args.dropout)
    fig_dir = FIG / root.name
    device = get_device()
    meta, _cubes = load_cubes(VARIETY)
    preds = load_predictions(meta, args.folds, root)
    metrics = compute_metrics(preds, device)

    (root / "metrics.md").write_text(to_markdown(metrics, args.folds))
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # The other designs appear in the fold comparison; these figures only earn their place for row folds.
    if args.folds == "row":
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_auc(metrics, fig_dir / "auc_comparison.png")
        fig_attention(preds, metrics, fig_dir / "attention_maps.png", device)
    log.info("[saved] results/%s/metrics.md", root.name)


if __name__ == "__main__":
    main()
