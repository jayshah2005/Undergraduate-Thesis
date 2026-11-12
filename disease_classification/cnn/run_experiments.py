"""Train the whole-leaf CNN (mean and attention pooling) on pooled varieties, fold by fold, beside the mean-spectrum baseline.

`run` is shared with ../within_variety_disease_cnn, which calls it once per variety.
"""
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
ROOT = HERE.parents[1]
for p in (ROOT, HERE.parent / "logistic_regression", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.backbones import BACKBONES, DEFAULT_BACKBONE, masked_norms  # noqa: E402
from helpers.data import load_cubes  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from logistic_regression import (BANDS, POOLS, ROWS, TASKS, band_indices, check_task, fold_of,  # noqa: E402
                                 fold_split, load_meta, load_means, results_name, run_baselines, task_labels,
                                 usable_leaves, varieties_of)
from model import LeafCNN  # noqa: E402
from helpers.pca_common import get_device, np_  # noqa: E402
from scoring import summary  # noqa: E402
from helpers.cnn_common import pad_to_canvas  # noqa: E402
from train import CANVAS, POOLING, TrainConfig, preprocess, train_fold  # noqa: E402

DEFAULT_POOL = "all"
DEFAULT_TASKS = ("leafroll", "red_blotch")
SEEDS = (0, 1, 2)
RESULTS, LOGS = HERE / "results", HERE / "logs"


def load_cube_list(variety: str) -> list[np.ndarray]:
    """Load the leaf cubes of a --variety choice, in the order of :func:`load_meta`.

    :param variety: A variety name, or a key of POOLS.
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


def load(variety: str, bands: str, device: torch.device) -> dict:
    """Load the chosen leaves onto the CANVAS × CANVAS canvas, plus labels, rows and leaf means, keeping the chosen bands.

    The mask comes from all 204 bands before any are dropped, so a leaf covers the same
    pixels whichever bands the network sees.

    n: leaves loaded (about 250 for one variety, 998 for all). d: bands kept (204 or 3). S: canvas side.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :param device: GPU or CPU.
    :type device: torch.device
    :return: meta, X (n, d, S, S) float16, mask (n, 1, S, S), S, means (n, d).
    :rtype: dict
    """
    X, mask, side = pad_to_canvas(load_cube_list(variety), device, CANVAS)
    if bands != "all":
        X = X[:, torch.as_tensor(band_indices(bands), device=device)]   # a copy; the 204-band canvas is freed
    return {"meta": load_meta(variety), "X": X, "mask": mask, "side": side,
            "means": load_means(variety, device, bands)}


def invariance_check(d: dict, backbone: str = DEFAULT_BACKBONE) -> dict:
    """Check that padding and position do not change a leaf's score.

    The same leaf goes through an untrained network in eval mode three ways: on
    its own canvas, shifted by 8 px, and on a canvas 32 px larger. With masked
    normalisation and masked pooling the logits must agree. Also checks that
    SNV leaves the background exactly 0.

    :param d: Output of :func:`load`.
    :type d: dict
    :param backbone: A key of BACKBONES.
    :type backbone: str
    :return: largest logit difference and whether the background stayed 0.
    :rtype: dict
    """
    torch.manual_seed(0)
    X, mask = d["X"][:4].float(), d["mask"][:4]
    x = preprocess(X, mask)
    out = {"max_logit_diff": 0.0, "background_zero": bool((x * (1 - mask)).abs().max() == 0)}
    for pooling in POOLING:
        model = LeafCNN(X.shape[1], pooling, backbone=backbone).to(X.device).eval()
        # Give the running statistics non-trivial values so eval mode is exercised.
        for bn in masked_norms(model):
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


def results_root(smoke: bool, dropout: float = TrainConfig.dropout, variety: str = DEFAULT_POOL,
                 augment: bool = True, bands: str = "all", results: Path = RESULTS,
                 backbone: str = DEFAULT_BACKBONE) -> Path:
    """Folder for one run's results, named like the baseline's (e.g. all, baco_chardonnay_vidal_rgb, vidal).

    Smoke runs, RGB-only runs, another dropout, runs without augmentation and other backbones are
    kept apart, e.g. all_rgb_no_augmentation or vidal_small_resnet. results is the results/ folder
    of the calling script.
    """
    name = results_name(variety, bands) + ("" if dropout == TrainConfig.dropout else f"_dropout{dropout:g}")
    name += "" if augment else "_no_augmentation"
    name += "" if backbone == DEFAULT_BACKBONE else f"_{backbone}"
    return results / "smoke" / name if smoke else results / name


def check_settings(root: Path, cfg: TrainConfig) -> None:
    """Refuse to resume into a folder whose runs used other training settings.

    Fold files already on disk are skipped, so without this check a rerun with
    changed settings would silently mix old and new runs.

    :param root: Results folder of this run.
    :type root: Path
    :param cfg: Training settings of this run.
    :type cfg: TrainConfig
    :raises RuntimeError: if root/config.json lists different training settings or another canvas.
    """
    if not (root / "config.json").exists():
        return
    old = json.loads((root / "config.json").read_text())
    if old.get("canvas") != CANVAS:
        raise RuntimeError(f"{root} holds runs on a {old.get('canvas')} px canvas, not {CANVAS}; "
                           "move or delete that folder first")
    not_training = ("seeds", "variety", "bands", "canvas")
    # A setting added after a run was made took its default value in that run.
    old_settings = {**asdict(TrainConfig()), **{k: v for k, v in old.items() if k not in not_training}}
    if old_settings != asdict(cfg):
        raise RuntimeError(f"{root} holds runs made with {old_settings}, not {asdict(cfg)}; "
                           "move or delete that folder first")


def run_dir(task: str, pooling: str, seed: int, root: Path) -> Path:
    """Folder holding one (task, pooling, seed)'s fold files."""
    return root / "runs" / task / pooling / f"seed{seed}"


def run_cnn(d: dict, fold_ids: np.ndarray, task: str, pooling: str, seed: int, fold: int,
            cfg: TrainConfig) -> dict:
    """Train and score one fold of one model; return what goes in the fold file.

    :param d: Output of :func:`load`.
    :type d: dict
    :param fold_ids: Fold per leaf from fold_of, length n.
    :type fold_ids: np.ndarray
    :param task: A key of TASKS.
    :type task: str
    :param pooling: "mean" or "attention".
    :type pooling: str
    :param seed: Repeat number 0-2.
    :type seed: int
    :param fold: Test fold 0-4.
    :type fold: int
    :param cfg: Training settings.
    :type cfg: TrainConfig
    :return: test and validation indices and probabilities, test pooling weights and training history.
    :rtype: dict
    """
    labels = task_labels(d["meta"], task)
    y = torch.tensor(labels, dtype=torch.float32, device=d["X"].device)
    split = fold_split(fold_ids, fold, usable_leaves(d["meta"], labels))
    t0 = time.time()
    r = train_fold(d["X"], d["mask"], y, split, pooling, 100 * seed + fold, cfg)
    return {
        "test_idx": split[2], "prob": np_(r["prob"]), "attention": np_(r["attention"]).astype(np.float16),
        "val_idx": split[1], "val_prob": np_(r["val_prob"]),
        "best_epoch": r["best_epoch"], "epochs": r["epochs"],
        "train_loss": np.array(r["train_loss"]), "val_loss": np.array(r["val_loss"]),
        "seconds": time.time() - t0,
    }


def load_model_runs(task: str, pooling: str, seeds: tuple, folds, n: int, root: Path) -> tuple:
    """Read one model's test probabilities and best epochs from its saved fold files.

    Reading the files, not this run's memory, means a resumed run still logs every score.

    :param task: A key of TASKS.
    :type task: str
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
            f = np.load(run_dir(task, pooling, seed, root) / f"fold{fold}.npz")
            probs[i, f["test_idx"]] = f["prob"]
            epochs.append(int(f["best_epoch"]))
    return probs, epochs


def format_scores(s: dict) -> str:
    """Write one model's scores on one log line.

    :param s: Output of :func:`scoring.summary`.
    :type s: dict
    :return: e.g. "R1 0.52 R2 0.61 ...; Baco 0.58 Vidal 0.60".
    :rtype: str
    """
    parts = [" ".join(f"{ROWS[k]} {a:.2f}" for k, a in s["by_fold"].items())]
    if s["by_variety"]:
        parts.append(" ".join(f"{v} {a:.2f}" for v, a in s["by_variety"].items()))
    return "; ".join(parts)


def report_command(args: argparse.Namespace) -> str:
    """The report.py command that gives this run's intervals, tables and figures.

    Another backbone's rows appear in the default run's report (SECOND_BACKBONE), so there is no --backbone flag.
    """
    flags = f"--variety {args.variety}"
    flags += " --bands rgb" if args.bands == "rgb" else ""
    flags += f" --dropout {args.dropout:g}" if args.dropout != TrainConfig.dropout else ""
    flags += " --no-augmentation" if args.no_augmentation else ""
    return f"report.py {flags}"


def run(variety: str, bands: str, tasks: list[str], root: Path, log, cfg: TrainConfig,
        seeds: tuple = SEEDS, folds=range(5)) -> None:
    """Check, then train every (task, pooling, seed, fold) of one variety choice not already on disk.

    Logs the baseline and each model's test balanced accuracy, one line each.

    :param variety: A variety name, or a key of POOLS.
    :type variety: str
    :param bands: A value of BANDS.
    :type bands: str
    :param tasks: Keys of TASKS, run one after the other.
    :type tasks: list[str]
    :param root: Results folder of this run, from :func:`results_root`.
    :type root: Path
    :param log: Logger from get_logger.
    :type log: logging.Logger
    :param cfg: Training settings.
    :type cfg: TrainConfig
    :param seeds: Seeds to train.
    :type seeds: tuple
    :param folds: Test folds to train.
    :type folds: Iterable[int]
    """
    # Check the tasks can run before anything is written or loaded onto the GPU
    meta = load_meta(variety)
    fold_ids = fold_of(meta)   # every leaf's fold 0-4 is its field row
    for task in tasks:
        check_task(meta, fold_ids, task)
    check_settings(root, cfg)   # a folder only resumes runs made with the same settings

    # Set up GPU
    torch.backends.cudnn.benchmark = False
    # warn_only: max-pool backward has no deterministic CUDA kernel; everything else is.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = get_device()

    # Load data
    d = load(variety, bands, device)

    # Log system information
    log.info("%s: %d leaves, %d bands, canvas %d", root.name, len(d["meta"]), d["X"].shape[1], d["side"])
    log.info("Balanced accuracy of the 3-seed ensemble (mean P >= 0.5) over the held-out rows (0.5 = chance). "
             "CNN and baseline train on the same 3 rows; checkpoints are chosen on a fourth, and test scores choose nothing.")

    chk = invariance_check(d, cfg.backbone)
    if chk["max_logit_diff"] > 1e-3 or not chk["background_zero"]:
        raise RuntimeError(f"padding leaks into the prediction: {chk}")

    # Baseline: logistic regression on the mean spectrum, from ../logistic_regression (seconds, so not saved)
    base = run_baselines(d["meta"], d["means"], tasks)

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({**asdict(cfg), "seeds": list(seeds), "variety": variety,
                                                  "bands": bands, "canvas": d["side"]}, indent=2))

    varieties = meta.variety.to_numpy()

    # Train the CNN
    for task in tasks:
        y = task_labels(meta, task)
        keep = usable_leaves(meta, y)

        # Log the task and the score to beat
        left_out = sorted(set(varieties[~keep]))
        base_scores = summary(base[task][keep][None], y[keep], fold_ids[keep], varieties[keep], device)
        log.info("%s (%d + / %d −%s): baseline bal. acc. %.3f; %s", task, y[keep].sum(), (1 - y[keep]).sum(),
                 f", left out {', '.join(left_out)}" if left_out else "", base_scores["bal_acc"],
                 format_scores(base_scores))

        for pooling in POOLING:
            seconds = 0.0
            for seed in seeds:

                # Get the output folder
                out = run_dir(task, pooling, seed, root)
                out.mkdir(parents=True, exist_ok=True)

                for fold in folds:
                    f = out / f"fold{fold}.npz"

                    # In case a run stopped part-way, skip the folds that are already done
                    if f.exists():
                        continue

                    # Train the CNN for the current fold
                    r = run_cnn(d, fold_ids, task, pooling, seed, fold, replace(cfg))
                    np.savez(f, **r)
                    seconds += r["seconds"]

            # Score the model on the leaves tested so far (only part of them in a smoke run)
            probs, epochs = load_model_runs(task, pooling, seeds, folds, len(y), root)
            tested = ~np.isnan(probs).any(0)
            s = summary(probs[:, tested], y[tested], fold_ids[tested], varieties[tested], device)
            log.info("%s | %s pooling: bal. acc. %.3f (%+.3f vs baseline); seeds %s; %s; median best epoch %.0f; %s",
                     task, pooling, s["bal_acc"], s["bal_acc"] - base_scores["bal_acc"],
                     " ".join(f"{a:.2f}" for a in s["seeds"]), format_scores(s), np.median(epochs),
                     f"{seconds:.0f} s" if seconds else "saved earlier")


def parse_args(choices: list[str], default, description: str) -> argparse.Namespace:
    """Parse the flags shared by this script and the within-variety runner.

    :param choices: Allowed --variety values.
    :type choices: list[str]
    :param default: Default --variety value (a list when several may be given).
    :type default: str | list
    :param description: What --variety means for the calling script.
    :type description: str
    :return: Parsed flags.
    :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", choices=choices, default=default,
                    nargs="+" if isinstance(default, list) else None, help=description)
    ap.add_argument("--bands", choices=BANDS, default="all",
                    help="all 204 bands, or only the 3 RGB bands; rgb adds _rgb to the folder name")
    ap.add_argument("--task", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS),
                    help="what counts as positive (see TASKS in logistic_regression.py); several run one after the other")
    ap.add_argument("--smoke", action="store_true", help="test row R1, seed 0, 3 epochs, into results/smoke/")
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout,
                    help="dropout before the last linear layer; a non-default value adds _dropout<value> to the folder")
    ap.add_argument("--no-augmentation", action="store_true",
                    help="train without rotations and flips; adds _no_augmentation to the folder name")
    ap.add_argument("--backbone", choices=list(BACKBONES), default=DEFAULT_BACKBONE,
                    help="the CNN before the pooling (helpers/backbones.py); another than the default adds "
                         "_<backbone> to the folder name")
    return ap.parse_args()


def run_settings(args: argparse.Namespace) -> tuple[TrainConfig, tuple, range]:
    """Training settings, seeds and folds for the parsed flags (a smoke run is 1 fold, 1 seed, 3 epochs)."""
    cfg = TrainConfig(max_epochs=3, patience=3) if args.smoke else TrainConfig()
    cfg = replace(cfg, dropout=args.dropout, augment=not args.no_augmentation, backbone=args.backbone)
    seeds, folds = ((0,), range(1)) if args.smoke else (SEEDS, range(5))
    return cfg, seeds, folds


def main() -> None:
    """Train one pool of varieties (default: all 998 leaves) and log each model's test balanced accuracy."""
    args = parse_args(list(POOLS), DEFAULT_POOL, "which varieties to pool into one model; results go to results/<pool>/")
    root = results_root(args.smoke, args.dropout, args.variety, not args.no_augmentation, args.bands,
                        backbone=args.backbone)
    log = get_logger(f"disease_cnn_{root.name}", log_file=None if args.smoke else LOGS / f"{root.name}.log")
    cfg, seeds, folds = run_settings(args)
    run(args.variety, args.bands, args.task, root, log, cfg, seeds, folds)
    log.info("[done] %s. For intervals, tables and figures: %s", root.relative_to(HERE), report_command(args))


if __name__ == "__main__":
    main()
