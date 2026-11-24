"""Train the six-head CNN for every (method, seed, fold): LS and MGDA, 3 seeds, 5 row folds.

The runs are set up to compare with the disease and variety CNNs: the same row folds (test row Rk,
validation row R(k+1), train on the other 3), the same seed rule (100 * seed + fold) and the same
3-seed ensemble, scored by report_moo.py.

The training code (data.py, model.py, moo.py, metrics.py, train.py) is used unchanged. This file
only loops over the runs, makes them deterministic and saves what ``train_fold`` computes but does
not return: each fold's test probabilities, the validation loss per epoch and, for MGDA, the
weights it gave each head.

Usage (from this folder)::

    ../.venv/bin/python -u run_moo.py --method ls      # 15 trainings, resumable
    ../.venv/bin/python -u run_moo.py --method mgda          # also mgda_gn, mgda_ln, mgda_lgn
    PYTHONPATH=../../torchjd/src ../.venv/bin/python -u run_moo.py --method svdpga   # also osnmfpga, upgrad (local torchjd)
    ../.venv/bin/python -u run_moo.py --method ls --smoke   # 1 seed x 5 folds x 1 epoch, results/smoke/
"""
from __future__ import annotations

import os

# Deterministic cuBLAS must be set before CUDA starts (same setting as the disease CNN runs).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402
from data import HEAD_NAMES, fold_split, get_device, load_vines  # noqa: E402
from train import TrainConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
from moo import TORCHJD_AGGREGATORS  # noqa: E402
from train import METHODS  # noqa: E402  (ls, mgda, mgda_gn, mgda_ln, mgda_lgn, svdpga, osnmfpga)
SEEDS = (0, 1, 2)
FOLDS = tuple(range(5))
CODE_FILES = ("data.py", "model.py", "moo.py", "metrics.py", "train.py")


def make_deterministic() -> None:
    """Same GPU settings as the disease CNN runs, so a rerun gives the same numbers."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def code_hashes(method: str = "ls") -> dict[str, str]:
    """sha256 of each training file, so the results can be tied to the code that made them.

    The PGA methods also record the torchjd they imported: its git commit, and whether its tree had changes.
    """
    hashes = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    if method in TORCHJD_AGGREGATORS:
        import torchjd
        repo = Path(torchjd.__file__).resolve().parents[2]
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        hashes["torchjd"] = f"{os.path.relpath(repo, HERE)} @ {git('rev-parse', 'HEAD')}" + (" (uncommitted changes)" if git("status", "--porcelain") else "")
    return hashes


class Recorder:
    """Capture values that ``train.train_fold`` computes but does not return.

    It temporarily wraps three functions that train.py calls by name:
    ``score_all`` (sees the test probabilities), ``aggregate_mgda`` and ``aggregate_torchjd`` (see
    each step's weights per head) and ``eval_total_loss`` (called once per epoch, so it marks the end of an epoch). The originals
    are restored on exit, and the wrapped functions return exactly what the originals return.
    """

    def __init__(self) -> None:
        self.prob: torch.Tensor | None = None   # (n_test, 6) test probabilities
        self.val_losses: list[float] = []        # summed validation loss per epoch
        self.epoch_alpha: list[torch.Tensor] = []  # MGDA / PGA: mean weight per head per epoch, each (6,)
        self._step_alpha: list[torch.Tensor] = []
        self._orig = (T.score_all, T.aggregate_mgda, T.aggregate_torchjd, T.eval_total_loss)

    def __enter__(self) -> "Recorder":
        score_all, aggregate_mgda, aggregate_torchjd, eval_total_loss = self._orig

        def wrapped_score_all(prob, y, varieties):
            self.prob = prob.detach().cpu().clone()
            return score_all(prob, y, varieties)

        def wrapped_aggregate_mgda(*args, **kwargs):
            g, alpha = aggregate_mgda(*args, **kwargs)
            self._step_alpha.append(alpha.detach().cpu())
            return g, alpha

        def wrapped_aggregate_torchjd(*args, **kwargs):
            g, weights = aggregate_torchjd(*args, **kwargs)
            self._step_alpha.append(weights.detach().cpu())
            return g, weights

        def wrapped_eval_total_loss(*args, **kwargs):
            # One call per epoch, right after its training steps: close that epoch's α average.
            if self._step_alpha:
                self.epoch_alpha.append(torch.stack(self._step_alpha).mean(0))
                self._step_alpha = []
            loss = eval_total_loss(*args, **kwargs)
            self.val_losses.append(loss)
            return loss

        T.score_all, T.aggregate_mgda, T.aggregate_torchjd, T.eval_total_loss = (
            wrapped_score_all, wrapped_aggregate_mgda, wrapped_aggregate_torchjd, wrapped_eval_total_loss)
        return self

    def __exit__(self, *exc) -> None:
        T.score_all, T.aggregate_mgda, T.aggregate_torchjd, T.eval_total_loss = self._orig


def check_config(out: Path, cfg: TrainConfig, method: str) -> None:
    """Write out/config.json, or check that an existing one matches.

    The training settings are shared by every method. The code hashes are kept per method, so a
    method added later (with moo.py extended) can join the folder, while resuming one method
    with changed code is refused (it must not mix code versions).

    :param out: Results folder.
    :param cfg: Training settings (method and seed are set per run).
    :param method: A value of METHODS.
    :raises SystemExit: if the settings, or this method's code, differ from the saved ones.
    """
    shared = {"train_config": {k: v for k, v in asdict(cfg).items() if k not in ("method", "seed")},
              "seed_rule": "100 * seed + fold", "seeds": list(SEEDS)}
    path = out / "config.json"
    have = json.loads(path.read_text()) if path.exists() else {
        **shared, "created": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
        "code_sha256_by_method": {}}
    for key, value in shared.items():
        if have.get(key) != value:
            raise SystemExit(f"{path}: '{key}' differs from the saved runs; use a new results folder.")
    by_method, now = have["code_sha256_by_method"], code_hashes(method)
    if (out / method).exists() and by_method.get(method, now) != now:
        raise SystemExit(f"{path}: the training code changed since the saved '{method}' runs; "
                         "use a new results folder.")
    by_method[method] = now
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(have, indent=1) + "\n")


def run_one(data, method: str, seed: int, fold: int, cfg: TrainConfig, device: torch.device,
            out: Path) -> dict | None:
    """Train one (method, seed, fold) and save its test predictions; skip it if already saved.

    :return: train_fold's result plus seconds, or None if the fold was already on disk.
    """
    path = out / method / f"seed{seed}" / f"fold{fold}.npz"
    if path.exists():
        return None
    run_cfg = replace(cfg, method=method, seed=100 * seed + fold)
    start = time.time()
    with Recorder() as rec:
        res = T.train_fold(data, fold, run_cfg, device)
    seconds = time.time() - start
    _tr, _va, te = fold_split(data.fold_ids, fold)
    alpha = torch.stack(rec.epoch_alpha).numpy() if rec.epoch_alpha else np.zeros((0, len(HEAD_NAMES)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, test_idx=te, paths=np.array(data.paths)[te], y=data.y[te], prob=rec.prob.numpy(),
             best_epoch=res["best_epoch"], epochs_run=res["epochs_run"], val_loss=res["val_loss"],
             val_losses=np.array(rec.val_losses), alpha=alpha, seconds=seconds, seed_used=run_cfg.seed)
    return {**res, "seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--smoke", action="store_true", help="1 seed x 5 folds x 1 epoch into results/smoke/")
    args = ap.parse_args()

    make_deterministic()
    device = get_device()
    cfg = TrainConfig()   # the defaults in train.py, as written
    seeds, out = SEEDS, RESULTS
    if args.smoke:
        cfg, seeds, out = replace(cfg, max_epochs=1), (0,), RESULTS / "smoke"
    check_config(out, cfg, args.method)

    print(f"loading vines on {device} ...", flush=True)
    data = load_vines()
    log = open(out / f"{args.method}.log", "a")
    for seed in seeds:
        for fold in FOLDS:
            res = run_one(data, args.method, seed, fold, cfg, device, out)
            if res is None:
                continue
            m = res["metrics"]
            line = (f"[{datetime.now():%F %T}] {args.method} seed {seed} fold {fold}: "
                    f"best epoch {res['best_epoch']} of {res['epochs_run']}, val loss {res['val_loss']:.3f}, "
                    f"{res['seconds']:.0f} s | variety bal. acc. {m['variety']['bal_acc']:.3f}, "
                    f"red blotch {m['disease']['RedBlotch']['bal_acc']:.3f}, "
                    f"leafroll {m['disease']['Leafroll']['bal_acc']:.3f}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print(f"[done] {out}/{args.method}. Score with report_moo.py", flush=True)


if __name__ == "__main__":
    main()
