"""Choose the CNN's epoch count inside each training split, then train and score one final model per seed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# run_experiments sets the CUDA determinism variable and sys.path, so it comes before torch here.
from run_experiments import (LOGS, RESULTS, ROWS, SEEDS, Config, TrainConfig, check_settings, fold_split,  # noqa: E402
                             get_device, get_logger, invariance_check, load, np_, train_fold)
from logistic_regression import BANDS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

EPOCH_CAP = 80
TRIAL_SEED = 1000     # one seed per trial; final models use the same seeds as the other runs


def trial_split(rows: np.ndarray, test_row: str, val_row: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train on the rows left once the test and validation rows are set aside.

    The third entry repeats the validation leaves: train_fold scores that set when it
    finishes, and a trial must never touch the test row.

    n: the number of leaves.

    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param test_row: The held-out test row, e.g. "R3".
    :type test_row: str
    :param val_row: The row this trial validates on.
    :type val_row: str
    :return: train, validation and validation leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(rows))
    val = idx[rows == val_row]
    return idx[(rows != test_row) & (rows != val_row)], val, val


def run_trials(d: dict, bands: torch.Tensor, rows: np.ndarray, k: int, tc: TrainConfig, out: Path) -> np.ndarray:
    """Run the 4 trials of one test row, skipping those already on disk.

    :param d: Output of :func:`run_experiments.load`.
    :type d: dict
    :param bands: Indices of the input bands, length d.
    :type bands: torch.Tensor
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param k: Test row number 0-4.
    :type k: int
    :param tc: Settings of a trial: tc.max_epochs epochs, early stopping switched off.
    :type tc: TrainConfig
    :param out: Folder of the trial files.
    :type out: Path
    :return: Validation loss per trial and epoch, shape (4, tc.max_epochs).
    :rtype: np.ndarray
    """
    out.mkdir(parents=True, exist_ok=True)
    curves = []
    for j, val_row in enumerate(r for r in ROWS if r != ROWS[k]):
        f = out / f"test{ROWS[k]}_val{val_row}.npz"
        if not f.exists():
            r = train_fold(d, bands, tc, trial_split(rows, ROWS[k], val_row), TRIAL_SEED + 10 * k + j)
            np.savez(f, val_loss=np.array(r["val_loss"]), train_loss=np.array(r["train_loss"]))
        curves.append(np.load(f)["val_loss"])
    return np.stack(curves)


def choose_epochs(curves: np.ndarray) -> int:
    """Apply the selection rule: the epoch with the lowest average validation loss.

    :param curves: Validation loss per trial and epoch, shape (4, epochs).
    :type curves: np.ndarray
    :return: The chosen epoch, counted from 1.
    :rtype: int
    """
    return int(curves.mean(0).argmin()) + 1


def run_finals(d: dict, bands: torch.Tensor, rows: np.ndarray, k: int, epochs: int, seeds: tuple, root: Path) -> None:
    """Train one final model per seed on all 4 non-test rows for exactly the chosen epochs; score the test row.

    :param d: Output of :func:`run_experiments.load`.
    :type d: dict
    :param bands: Indices of the input bands, length d.
    :type bands: torch.Tensor
    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param k: Test row number 0-4.
    :type k: int
    :param epochs: The chosen epoch count.
    :type epochs: int
    :param seeds: Seeds of the final models.
    :type seeds: tuple
    :param root: Results folder of this run.
    :type root: Path
    """
    split = fold_split(rows, k, validation=False)
    tc = TrainConfig(max_epochs=epochs, validation=False)
    for seed in seeds:
        out = root / f"seed{seed}"
        out.mkdir(exist_ok=True)
        f = out / f"fold{k}.npz"
        if f.exists():
            continue
        r = train_fold(d, bands, tc, split, 100 * seed + k)
        np.savez(f, test_idx=split[2], prob=np_(r["prob"]), best_epoch=r["best_epoch"], epochs=r["epochs"],
                 train_loss=np.array(r["train_loss"]), val_loss=np.array(r["val_loss"]))


def main() -> None:
    """Check, then for each test row: trials → choose the epoch count → final models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=BANDS, default=Config.bands, help="all 204 bands, or only the 3 RGB bands")
    ap.add_argument("--smoke", action="store_true", help="first test row, 2-epoch trials, 1 final seed, results/smoke/")
    args = ap.parse_args()

    cap, seeds, tests = (2, (0,), (0,)) if args.smoke else (EPOCH_CAP, SEEDS, range(5))
    # A patience above the cap means a trial always runs to the cap.
    tc_trial = TrainConfig(max_epochs=cap, patience=cap + 1)
    run_name = f"{args.bands}_nested"
    root = (RESULTS / "smoke" if args.smoke else RESULTS) / run_name
    log = get_logger("variety_cnn_nested", log_file=None if args.smoke else LOGS / f"{run_name}.log")
    check_settings(root, tc_trial)

    torch.backends.cudnn.benchmark = False
    # warn_only: max-pool backward has no deterministic CUDA kernel; everything else is.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = get_device()

    d = load(device)
    bands = torch.as_tensor(rgb_bands_of(d, args.bands), device=device)
    rows = d["meta"].row.to_numpy().astype(str)
    log.info("%s: %d leaves, %d bands, epoch cap %d", run_name, len(rows), len(bands), cap)
    change = invariance_check(d, bands)
    if change > 1e-3:
        raise RuntimeError(f"padding leaks into the prediction: {change}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"bands": args.bands, "training": asdict(tc_trial),
                                                  "seeds": list(seeds), "canvas": d["side"]}, indent=2))
    chosen = {}
    for k in tests:
        t0 = time.time()
        epochs = choose_epochs(run_trials(d, bands, rows, k, tc_trial, root / "trials"))
        chosen[ROWS[k]] = epochs
        # Written before the final models, so the choice is on disk before any test score exists.
        (root / "chosen_epochs.json").write_text(json.dumps(chosen, indent=2))
        run_finals(d, bands, rows, k, epochs, seeds, root)
        log.info("test %s: %d epochs chosen, %.0f s", ROWS[k], epochs, time.time() - t0)
    log.info("[done] %s", root.relative_to(HERE))


def rgb_bands_of(d: dict, bands: str) -> np.ndarray:
    """Indices of the input bands.

    :param d: Output of :func:`run_experiments.load`.
    :type d: dict
    :param bands: "all" or "rgb".
    :type bands: str
    :return: Band indices, length d.
    :rtype: np.ndarray
    """
    return d["rgb"] if bands == "rgb" else np.arange(d["X"].shape[1])


if __name__ == "__main__":
    main()
