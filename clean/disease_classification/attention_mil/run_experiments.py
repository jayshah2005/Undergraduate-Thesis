"""Train the whole-leaf CNN arms and the mean-spectrum baseline, fold by fold, for chosen varieties and tasks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

# Needed by cuBLAS for deterministic matrix products; must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data import load_cubes, load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from model import LeafCNN  # noqa: E402
from pca_common import VARIETIES, get_device, np_  # noqa: E402
from scoring import auc_groups, auc_summary  # noqa: E402
from train import (BRIGHTNESS, POOLING, ROWS, TASKS, TrainConfig, baseline_fold, check_task,  # noqa: E402
                   fold_of, fold_split, pad_to_canvas, preprocess, task_labels, train_fold, usable_leaves)

DEFAULT_VARIETY, ALL_VARIETIES = "Chardonnay", "all"
DEFAULT_TASKS = ("leafroll", "red_blotch")
SEEDS = (0, 1, 2)
RESULTS, LOGS = HERE / "results", HERE / "logs"


def varieties_of(variety: str) -> list[str]:
    """Varieties covered by a --variety choice.

    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :return: Variety names.
    :rtype: list[str]
    """
    return list(VARIETIES) if variety == ALL_VARIETIES else [variety]


def load_meta(variety: str):
    """Load the leaf table of a --variety choice, without the cubes.

    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :return: Leaf table (path, variety, row, vine, labels) with a fresh index.
    :rtype: pandas.DataFrame
    """
    meta = load_dataset()[0]
    return meta[meta.variety.isin(varieties_of(variety))].reset_index(drop=True)


def load_cube_list(variety: str) -> list[np.ndarray]:
    """Load the leaf cubes of a --variety choice, in the order of :func:`load_meta`.

    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :return: One cube (204, lines, samples) per leaf.
    :rtype: list[np.ndarray]
    """
    paths, cubes = [], []
    for v in varieties_of(variety):
        m, c = load_cubes(v)
        paths += list(m.path)
        cubes += c
    assert paths == list(load_meta(variety).path), "cube and leaf-table order differ"
    return cubes


def load(variety: str, device: torch.device) -> dict:
    """Load the chosen leaves onto one canvas, plus labels, rows and leaf means.

    n: leaves loaded (250 for one variety, 998 for all). S: canvas side.

    :param variety: A variety name, or ALL_VARIETIES.
    :type variety: str
    :param device: GPU or CPU.
    :type device: torch.device
    :return: meta, X (n, 204, S, S) float16, mask (n, 1, S, S), S, means (n, 204).
    :rtype: dict
    """
    X, mask, side = pad_to_canvas(load_cube_list(variety), device)
    all_meta, _w, all_means, _g = load_dataset()
    keep = all_meta.variety.isin(varieties_of(variety)).to_numpy()
    return {"meta": load_meta(variety), "X": X, "mask": mask, "side": side,
            "means": torch.tensor(all_means[keep], dtype=torch.float64, device=device)}


def invariance_check(d: dict) -> dict:
    """Check that padding and position do not change a leaf's score.

    The same leaf goes through an untrained network in eval mode three ways: on
    its own canvas, shifted by 8 px, and on a canvas 32 px larger. With masked
    normalisation and masked pooling the logits must agree. Also checks that
    both preprocessings leave the background exactly 0.

    :param d: Output of :func:`load`.
    :type d: dict
    :return: largest logit difference and whether the background stayed 0.
    :rtype: dict
    """
    torch.manual_seed(0)
    X, mask = d["X"][:4].float(), d["mask"][:4]
    stats = (X.mean((0, 2, 3)), X.std((0, 2, 3)) + 1.0)
    g = torch.Generator(device=X.device).manual_seed(0)
    cfg = TrainConfig()
    out = {"max_logit_diff": 0.0, "background_zero": True}
    for brightness in BRIGHTNESS:
        x = preprocess(X, mask, brightness, stats, True, cfg, g)
        out["background_zero"] &= bool((x * (1 - mask)).abs().max() == 0)
        for pooling in POOLING:
            model = LeafCNN(X.shape[1], pooling).to(X.device).eval()
            # Give the running statistics non-trivial values so eval mode is exercised.
            for bn in model.bn:
                bn.running_mean.normal_()
                bn.running_var.uniform_(0.5, 2.0)
            pad = torch.nn.functional.pad
            with torch.no_grad():
                base, _ = model(x, mask)
                # (left, right, top, bottom): shift by 8 px, and a 32 px larger canvas.
                diffs = [(model(pad(x, p), pad(mask, p))[0] - base).abs().max().item()
                         for p in ((8, 0, 8, 0), (16, 16, 16, 16), (0, 32, 0, 32))]
            out["max_logit_diff"] = max(out["max_logit_diff"], *diffs)
    return out


def variant_suffix(brightness_jitter: bool = True, dropout: float = TrainConfig.dropout) -> str:
    """Folder-name suffix of a run whose settings differ from the defaults ("" if none do)."""
    suffix = "" if brightness_jitter else "_no_brightness_jitter"
    return suffix if dropout == TrainConfig.dropout else f"{suffix}_dropout{dropout:g}"


def variety_prefix(variety: str = DEFAULT_VARIETY) -> str:
    """Folder-name prefix of a variety choice ("" for the default, e.g. "cab_franc_")."""
    return "" if variety == DEFAULT_VARIETY else variety.lower().replace(" ", "_") + "_"


def results_root(smoke: bool, brightness_jitter: bool = True, dropout: float = TrainConfig.dropout,
                 variety: str = DEFAULT_VARIETY) -> Path:
    """Folder for one run's results (smoke runs, varieties and other settings kept apart)."""
    name = variety_prefix(variety) + "row" + variant_suffix(brightness_jitter, dropout)
    return RESULTS / "smoke" / name if smoke else RESULTS / name


def check_settings(root: Path, cfg: TrainConfig) -> None:
    """Refuse to resume into a folder whose runs used other training settings.

    Fold files already on disk are skipped, so without this check a rerun with
    changed settings would silently mix old and new runs.

    :param root: Results folder of this run.
    :type root: Path
    :param cfg: Training settings of this run.
    :type cfg: TrainConfig
    :raises RuntimeError: if root/config.json lists different training settings.
    """
    if not (root / "config.json").exists():
        return
    old = json.loads((root / "config.json").read_text())
    not_training = ("seeds", "variety", "canvas")
    # A setting added after a run was made took its default value in that run.
    old_settings = {**asdict(TrainConfig()), **{k: v for k, v in old.items() if k not in not_training}}
    if old_settings != asdict(cfg):
        raise RuntimeError(f"{root} holds runs made with {old_settings}, not {asdict(cfg)}; "
                           "move or delete that folder first")


def run_baselines(d: dict, fold: np.ndarray, tasks: list[str]) -> dict:
    """Score the mean-spectrum baseline for the chosen tasks, on the rows the CNN trains on.

    It fits on the CNN's 3 training rows, so neither model has more data. It has no setting to
    choose, so the validation row stays unused.

    :param d: Output of :func:`load`.
    :type d: dict
    :param fold: Fold per leaf from fold_of, length n.
    :type fold: np.ndarray
    :param tasks: Keys of TASKS.
    :type tasks: list[str]
    :return: {task: P(positive) per leaf, length n; 0 for leaves the task leaves out}.
    :rtype: dict
    """
    out = {}
    for task in tasks:
        y = task_labels(d["meta"], task)
        keep = usable_leaves(d["meta"], y)
        p = np.zeros(len(y))
        for k in range(5):
            tr, _va, te = fold_split(fold, k, keep)
            p[te] = baseline_fold(d["means"], y, tr, te)
        out[task] = p
    return out


def save_baselines(path: Path, base: dict) -> None:
    """Write the baselines, keeping those of other tasks already in the file.

    :param path: The results folder's baseline.npz.
    :type path: Path
    :param base: Output of :func:`run_baselines`.
    :type base: dict
    """
    old = dict(np.load(path)) if path.exists() else {}
    np.savez(path, **{**old, **base})


def run_dir(task: str, brightness: str, pooling: str, seed: int, root: Path) -> Path:
    """Folder holding one (task, arm, seed)'s fold files."""
    return root / "runs" / task / f"{brightness}__{pooling}" / f"seed{seed}"


def run_cnn(d: dict, fold_ids: np.ndarray, task: str, brightness: str, pooling: str, seed: int,
            fold: int, cfg: TrainConfig) -> dict:
    """Train and score one fold of one arm; return what goes in the fold file.

    :param d: Output of :func:`load`.
    :type d: dict
    :param fold_ids: Fold per leaf from fold_of, length n.
    :type fold_ids: np.ndarray
    :param task: A key of TASKS.
    :type task: str
    :param brightness: "band_z" or "snv".
    :type brightness: str
    :param pooling: "mean" or "attention".
    :type pooling: str
    :param seed: Repeat number 0-2.
    :type seed: int
    :param fold: Test fold 0-4.
    :type fold: int
    :param cfg: Training settings.
    :type cfg: TrainConfig
    :return: test indices, probabilities, pooling weights and training history.
    :rtype: dict
    """
    labels = task_labels(d["meta"], task)
    y = torch.tensor(labels, dtype=torch.float32, device=d["X"].device)
    split = fold_split(fold_ids, fold, usable_leaves(d["meta"], labels))
    t0 = time.time()
    r = train_fold(d["X"], d["mask"], y, split, brightness, pooling, 100 * seed + fold, cfg)
    return {
        "test_idx": split[2], "prob": np_(r["prob"]), "attention": np_(r["attention"]).astype(np.float16),
        "best_epoch": r["best_epoch"], "epochs": r["epochs"],
        "train_loss": np.array(r["train_loss"]), "val_loss": np.array(r["val_loss"]),
        "seconds": time.time() - t0,
    }


def load_arm(task: str, brightness: str, pooling: str, seeds: tuple, folds, n: int, root: Path) -> tuple:
    """Read one arm's test probabilities and best epochs from its saved fold files.

    Reading the files, not this run's memory, means a resumed run still logs every score.

    :param task: A key of TASKS.
    :type task: str
    :param brightness: "band_z" or "snv".
    :type brightness: str
    :param pooling: "mean" or "attention".
    :type pooling: str
    :param seeds: Seeds of this run.
    :type seeds: tuple
    :param folds: Test folds of this run.
    :type folds: Iterable[int]
    :param n: Leaves loaded.
    :type n: int
    :param root: Results folder of this run.
    :type root: Path
    :return: P(positive) per seed and leaf, shape (len(seeds), n), NaN for leaves not tested, and the best epochs.
    :rtype: tuple
    """
    probs, epochs = np.full((len(seeds), n), np.nan), []
    for i, seed in enumerate(seeds):
        for fold in folds:
            f = np.load(run_dir(task, brightness, pooling, seed, root) / f"fold{fold}.npz")
            probs[i, f["test_idx"]] = f["prob"]
            epochs.append(int(f["best_epoch"]))
    return probs, epochs


def format_scores(s: dict) -> str:
    """Write one model's scores on one log line.

    :param s: Output of :func:`scoring.auc_summary`.
    :type s: dict
    :return: e.g. "R1 0.52 R2 0.61 ...; Baco 0.58 Vidal 0.60".
    :rtype: str
    """
    parts = [" ".join(f"{ROWS[k]} {a:.2f}" for k, a in s["by_fold"].items())]
    if s["by_variety"]:
        parts.append(" ".join(f"{v} {a:.2f}" for v, a in s["by_variety"].items()))
    return "; ".join(parts)


def report_command(args: argparse.Namespace) -> str:
    """The report.py command that gives this run's intervals, tables and figures."""
    variety = f'"{args.variety}"' if " " in args.variety else args.variety
    flags = f"--variety {variety}"
    flags += " --no-brightness-jitter" if args.no_brightness_jitter else ""
    flags += f" --dropout {args.dropout:g}" if args.dropout != TrainConfig.dropout else ""
    return f"report.py {flags}"


def main() -> None:
    """Check, then train every (task, arm, seed, fold) not already on disk, logging each arm's test AUC."""

    # parse arguments
    # --variety: one variety, or all to pool them (default: Chardonnay)
    # --task: one or more of leafroll, red_blotch, both, any (default: leafroll and red_blotch, each on its own)
    # --smoke: test row R1, seed 0, 3 epochs, into results/smoke/
    # --no-brightness-jitter: band_z arms train without the random brightness scale,
    #                         into results/row_no_brightness_jitter/
    # --dropout: dropout before the last linear layer; a non-default value adds _dropout<value> to the folder name
    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", choices=[*VARIETIES, ALL_VARIETIES], default=DEFAULT_VARIETY,
                    help="one variety, or all to pool them; other than Chardonnay, the folder name starts with it")
    ap.add_argument("--task", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS),
                    help="what counts as positive (see TASKS in train.py); several run one after the other")
    ap.add_argument("--smoke", action="store_true", help="test row R1, seed 0, 3 epochs, into results/smoke/")
    ap.add_argument("--no-brightness-jitter", action="store_true",
                    help="turn off the band_z arms' brightness jitter; results go to row_no_brightness_jitter/")
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout,
                    help="dropout before the last linear layer; a non-default value adds _dropout<value> to the folder")
    args = ap.parse_args()

    # Check the tasks can run before anything is written or loaded onto the GPU
    meta = load_meta(args.variety)
    fold_ids = fold_of(meta)   # every leaf's fold 0-4 is its field row
    for task in args.task:
        check_task(meta, fold_ids, task)

    # Set up logging and root file
    root = results_root(args.smoke, not args.no_brightness_jitter, args.dropout, args.variety)
    log = get_logger("attention_mil", log_file=None if args.smoke else LOGS / f"{root.name}.log")

    # Set up GPU
    torch.backends.cudnn.benchmark = False
    # warn_only: max-pool backward has no deterministic CUDA kernel; everything else is.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = get_device()

    # Set up training config
    cfg = TrainConfig(max_epochs=3, patience=3) if args.smoke else TrainConfig()
    if args.no_brightness_jitter:
        cfg = replace(cfg, brightness_jitter=0.0)
    cfg = replace(cfg, dropout=args.dropout)
    seeds, folds = ((0,), (0,)) if args.smoke else (SEEDS, range(5))
    check_settings(root, cfg)   # check if the root folder is empty or contains runs with different settings

    # Load data
    d = load(args.variety, device)

    # Log system information
    log.info("%s: %d leaves, canvas %d", root.name, len(d["meta"]), d["side"])
    log.info("AUC = 3-seed ensemble on held-out rows, comparing only leaves of the same variety and row (0.5 = chance). "
             "CNN and baseline train on the same 3 rows; checkpoints are chosen on a fourth, and test scores choose nothing.")

    chk = invariance_check(d)
    if chk["max_logit_diff"] > 1e-3 or not chk["background_zero"]:
        raise RuntimeError(f"padding leaks into the prediction: {chk}")

    # Baseline is a logistic regression on the mean spectrum
    base = run_baselines(d, fold_ids, args.task)

    # Save baseline results
    root.mkdir(parents=True, exist_ok=True)
    save_baselines(root / "baseline.npz", base)

    (root / "config.json").write_text(json.dumps({**asdict(cfg), "seeds": list(seeds), "variety": args.variety,
                                                  "canvas": d["side"]}, indent=2))

    groups = auc_groups(meta)
    varieties = meta.variety.to_numpy()

    # Train the CNN
    for task in args.task:
        y = task_labels(meta, task)
        keep = usable_leaves(meta, y)

        # Log the task and the score to beat
        left_out = sorted(set(varieties[~keep]))
        base_scores = auc_summary(base[task][keep][None], y[keep], groups[keep], fold_ids[keep],
                                  varieties[keep], device)
        log.info("%s (%d + / %d −%s): baseline AUC %.3f; %s", task, y[keep].sum(), (1 - y[keep]).sum(),
                 f", left out {', '.join(left_out)}" if left_out else "", base_scores["auc"],
                 format_scores(base_scores))

        for brightness in BRIGHTNESS:
            for pooling in POOLING:
                seconds = 0.0
                for seed in seeds:

                    # Get the output folder
                    out = run_dir(task, brightness, pooling, seed, root)
                    out.mkdir(parents=True, exist_ok=True)

                    for fold in folds:
                        f = out / f"fold{fold}.npz"

                        # Incase there is an error and we want to resume training, we skip the folds that are already done
                        if f.exists():
                            continue

                        # Train the CNN for the current fold
                        r = run_cnn(d, fold_ids, task, brightness, pooling, seed, fold, replace(cfg))
                        np.savez(f, **r)
                        seconds += r["seconds"]

                # Score the arm on the leaves tested so far (only part of them in a smoke run)
                probs, epochs = load_arm(task, brightness, pooling, seeds, folds, len(y), root)
                tested = ~np.isnan(probs).any(0)
                s = auc_summary(probs[:, tested], y[tested], groups[tested], fold_ids[tested], varieties[tested], device)
                log.info("%s | %s + %s: AUC %.3f (%+.3f vs baseline); seeds %s; %s; median best epoch %.0f; %s",
                         task, brightness, pooling, s["auc"], s["auc"] - base_scores["auc"],
                         " ".join(f"{a:.2f}" for a in s["seeds"]), format_scores(s), np.median(epochs),
                         f"{seconds:.0f} s" if seconds else "saved earlier")

    log.info("[done] %s. For intervals, tables and figures: %s", root.relative_to(HERE), report_command(args))


if __name__ == "__main__":
    main()
