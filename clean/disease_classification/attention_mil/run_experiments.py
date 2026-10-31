"""Train the whole-leaf CNN arms and the mean-spectrum baseline on Chardonnay, fold by fold."""
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
from pca_common import get_device, np_  # noqa: E402
from train import (BRIGHTNESS, FOLD_SCHEMES, POOLING, TASKS, TrainConfig, baseline_fold,  # noqa: E402
                   fold_of, fold_split, pad_to_canvas, preprocess, train_fold)

VARIETY = "Chardonnay"
SEEDS = (0, 1, 2)
RESULTS, LOGS = HERE / "results", HERE / "logs"


def load(device: torch.device) -> dict:
    """Load the variety's cubes onto one canvas, plus labels, rows and leaf means.

    n: leaves of the variety (250). S: canvas side.

    :param device: GPU or CPU.
    :type device: torch.device
    :return: meta, X (n, 204, S, S) float16, mask (n, 1, S, S), S, means (n, 204).
    :rtype: dict
    """
    meta, cubes = load_cubes(VARIETY)
    X, mask, side = pad_to_canvas(cubes, device) 
    all_meta, _w, all_means, _g = load_dataset()
    keep = (all_meta.variety == VARIETY).to_numpy()
    # Both tables are sorted by row and vine; check before pairing leaf i with mean i.
    assert list(all_meta.path[keep]) == list(meta.path), "cube and mean-spectrum order differ"
    return {"meta": meta, "X": X, "mask": mask, "side": side,
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


def results_root(scheme: str, smoke: bool, brightness_jitter: bool = True,
                 dropout: float = TrainConfig.dropout) -> Path:
    """Folder for one fold scheme's results (smoke runs and other settings kept apart)."""
    name = scheme + variant_suffix(brightness_jitter, dropout)
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
    not_training = ("seeds", "variety", "canvas", "folds", "folds_meaning")
    # A setting added after a run was made took its default value in that run.
    old_settings = {**asdict(TrainConfig()), **{k: v for k, v in old.items() if k not in not_training}}
    if old_settings != asdict(cfg):
        raise RuntimeError(f"{root} holds runs made with {old_settings}, not {asdict(cfg)}; "
                           "move or delete that folder first")


def run_baselines(d: dict, fold: np.ndarray) -> dict:
    """Score the mean-spectrum baseline for both tasks on the same folds as the CNN.

    Two versions: trained on all 4 non-test folds (the normal way to use it), and on
    the same 3 folds the CNN trains on (the CNN gives its 4th fold to early stopping).
    The keys keep the names "4rows" / "3rows" from the row scheme.

    :param d: Output of :func:`load`.
    :type d: dict
    :param fold: Fold per leaf from fold_of, length n.
    :type fold: np.ndarray
    :return: {task: {"4rows" | "3rows": P(positive) per leaf, length n}}.
    :rtype: dict
    """
    out = {}
    for task, col in TASKS.items():
        y = d["meta"][col].to_numpy()
        p4, p3 = np.zeros(len(y)), np.zeros(len(y))
        for k in range(5):
            tr, va, te = fold_split(fold, k)
            p4[te] = baseline_fold(d["means"], y, np.concatenate([tr, va]), te) # train on all 4 non-test folds (the normal way to use it)
            p3[te] = baseline_fold(d["means"], y, tr, te) # train on the same 3 folds the CNN trains on (the CNN gives its 4th fold to early stopping)
        out[task] = {"4rows": p4, "3rows": p3}
    return out


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
    :param task: "leafroll" or "red_blotch".
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
    y = torch.tensor(d["meta"][TASKS[task]].to_numpy(), dtype=torch.float32, device=d["X"].device)
    split = fold_split(fold_ids, fold)
    t0 = time.time()
    r = train_fold(d["X"], d["mask"], y, split, brightness, pooling, 100 * seed + fold, cfg)
    return {
        "test_idx": split[2], "prob": np_(r["prob"]), "attention": np_(r["attention"]).astype(np.float16),
        "best_epoch": r["best_epoch"], "epochs": r["epochs"],
        "train_loss": np.array(r["train_loss"]), "val_loss": np.array(r["val_loss"]),
        "seconds": time.time() - t0,
    }


def main() -> None:
    """Check, then train every (task, arm, seed, fold) not already on disk."""

    # parse arguments
    # --folds: how leaves are split into folds 
    #           choices: row, photo (random photos from every row), block (default: row)
    # --smoke: fold 0, seed 0, 3 epochs, into results/smoke/<folds>/
    # --no-brightness-jitter: band_z arms train without the random brightness scale,
    #                         into results/<folds>_no_brightness_jitter/
    # --dropout: dropout before the last linear layer; a non-default value adds _dropout<value> to the folder name
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", choices=list(FOLD_SCHEMES), default="row",
                    help="how leaves are split into folds (see FOLD_SCHEMES in train.py)")
    ap.add_argument("--smoke", action="store_true", help="fold 0, seed 0, 3 epochs, into results/smoke/<folds>/")
    ap.add_argument("--no-brightness-jitter", action="store_true",
                    help="turn off the band_z arms' brightness jitter; results go to <folds>_no_brightness_jitter/")
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout,
                    help="dropout before the last linear layer; a non-default value adds _dropout<value> to the folder")
    args = ap.parse_args()

    # Set up logging and root file
    root = results_root(args.folds, args.smoke, not args.no_brightness_jitter, args.dropout)
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
    d = load(device)
    fold_ids = fold_of(d["meta"], args.folds)   # give every leaf its fold 0-4 under one fold scheme


    # Log system information
    log.info("%s: %d leaves, canvas %d", root.name, len(d["meta"]), d["side"])

    chk = invariance_check(d)
    if chk["max_logit_diff"] > 1e-3 or not chk["background_zero"]:
        raise RuntimeError(f"padding leaks into the prediction: {chk}")

    # Baseline is a logistic regression on the mean spectrum
    base = run_baselines(d, fold_ids)

    # Save baseline results
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "baseline.npz", **{f"{t}__{k}": v for t, b in base.items() for k, v in b.items()})

    (root / "config.json").write_text(json.dumps({**asdict(cfg), "seeds": list(seeds), "variety": VARIETY,
                                                  "canvas": d["side"], "folds": args.folds}, indent=2))

    # Train the CNN
    for task in TASKS:
        for brightness in BRIGHTNESS:
            for pooling in POOLING:
                epochs, seconds = [], 0.0
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

                        epochs.append(r["best_epoch"])
                        seconds += r["seconds"]

                if epochs:
                    log.info("%s | %s + %s: %d trainings, median best epoch %.0f, %.0f s", task, brightness, pooling,
                             len(epochs), np.median(epochs), seconds)

    log.info("[done] %s", root.relative_to(HERE))


if __name__ == "__main__":
    main()
