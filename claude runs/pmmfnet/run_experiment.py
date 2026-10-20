"""Run your own PMMFNet experiments from a config file.

Run from claude runs/pmmfnet/:
    ../../.venv/bin/python -u run_experiment.py configs/my_run.toml                 # 5 folds -> runs/custom/my_run/
    ../../.venv/bin/python -u run_experiment.py configs/my_run.toml --smoke         # ~1 min check -> runs/custom/smoke/my_run/
    ../../.venv/bin/python -u run_experiment.py configs/my_run.toml --set training.lr=3e-5 --set name=lr3e-5
    ../../.venv/bin/python -u run_experiment.py configs/a.toml configs/b.toml       # one after another
    ../../.venv/bin/python -u run_experiment.py --summary                           # one table of all custom runs

Copy configs/template.toml: it lists every setting with its default. A config only needs the
settings it changes; the rest take the defaults, which reproduce controlled experiment A2 (the
paper's training with plan.md's protocol). Unknown settings or wrong types stop the run before
it starts. --set takes TOML values (numbers, true/false, [lists], "strings"; a bare word is
read as a string) and is applied after the config file.

Each run writes runs/custom/<name>/: <name>_oof.csv (leaf probabilities + prediction),
<name>_metrics.md/.json (confusion matrix, balanced accuracy, macro-F1, per-class
precision/recall, ROC AUC pooled and within-variety, per disease, per variety, per fold),
config.json (the resolved settings), folds/, checkpoints/fold<k>.pt, fold_assignment.csv, log.txt.
Everything shown on the terminal during a run is also appended to its log.txt (the shared logger,
../shared/logs.py), together with the traceback if the run fails.
Runs resume: finished folds are reloaded when the saved config matches; a different config under
an existing name is refused (choose another name, or pass --overwrite to delete the old run).

Terms: fold k = test row k of every variety; its validation row is row (k+1) mod 5; bal. acc. =
4-class balanced accuracy of the leaf predictions; MIL = the leaf-level loss of controlled L2.

Not configurable (built into 11_pmmfnet.py, which stays unchanged): the data, the 4 classes, the
5 row folds with validation row (k+1) mod 5, 5x5 interior patches, 204 bands zero-filled to
208, Adam, bf16 autocast, leaf probability = mean softmax over the leaf's patches.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # needed by deterministic = true

import argparse                                                     # noqa: E402
import copy                                                         # noqa: E402
import importlib                                                    # noqa: E402
import io                                                           # noqa: E402
import json                                                         # noqa: E402
import logging                                                      # noqa: E402
import re                                                           # noqa: E402
import shutil                                                       # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
import tomllib                                                      # noqa: E402
import traceback                                                    # noqa: E402
from contextlib import contextmanager, redirect_stdout              # noqa: E402
from pathlib import Path                                            # noqa: E402

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
from sklearn.metrics import balanced_accuracy_score                 # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))               # common.py, torch_common.py, four_class.py, logs.py
from common import N_FOLDS, fold_indices, load_meta                 # noqa: E402
from four_class import CLASSES, fmt, leaf_classes, save_results     # noqa: E402
from logs import close_log_file, get_logger                         # noqa: E402
from torch_common import seed_everything                            # noqa: E402
import pmmfnet                                                      # noqa: E402

m11 = importlib.import_module("11_pmmfnet")                         # the original pipeline's helpers

RUNS = HERE / "runs" / "custom"
EVAL_PERM_PATH = HERE / "cache" / "eval_permutations_seed12345.npy"   # shared with controlled.py (S2)
EVAL_PERM_SEED = 12345
VAL_PATCH_SEED = 1                  # fixed validation patches, the same whatever `seed` is
TEST_PATCH_SEED = 2                 # only used when evaluation.test_patches_per_leaf > 0

# Every setting and its default (= controlled experiment A2). configs/template.toml mirrors this.
# The types of these defaults are also the type check for config files and --set.
DEFAULTS = {
    "name": "",                     # output folder runs/custom/<name>; empty = the config file's name
    "description": "",
    "folds": [0, 1, 2, 3, 4],       # test folds to run (row R(k+1) of every variety)
    "seed": 0,                      # fold k trains with seed + k (0 = the controlled experiments)
    "deterministic": False,         # true: deterministic cuDNN kernels (slower, more reproducible)
    "data": {
        "scaler": "train",          # global min/max fitted on "train" leaves or "train+val" (original)
    },
    "model": {
        "channels": 32,
        "p_kernels": [7, 7, 4, 3, 3, 3],    # P (NAM) branch convs; the list length = number of convs
        "i_kernels": [7, 5, 4, 3, 3, 3],    # I (CBAM) branch, each conv followed by CBAM
        "d_kernels": [5, 5, 5, 5, 4, 3],    # D (GAM) branch
    },
    "training": {
        "objective": "patch",       # "patch" = cross-entropy per patch; "leaf" = leaf-level MIL loss (L2)
        "shuffle": True,            # spatial shuffle of the 24 neighbours of every training patch
        "lr": 1e-5,
        "weight_decay": 0.0,        # Adam's L2 penalty
        "batch_size": 256,          # patch inputs per optimizer step
        "samples_per_class": 100_000,       # patch inputs per class per epoch
        "leaves_per_class": 2,      # objective = "leaf": leaves per class in each step
        "patches_per_leaf": 32,     # objective = "leaf": patches per leaf; 4 x the two = batch_size
    },
    "early_stopping": {
        "metric": "balanced_accuracy",      # "balanced_accuracy" (ties -> lower loss) or "loss" (original)
        "patience": 5,              # epochs without improvement of the metric
        "max_epochs": 50,
        "min_delta": 0.0,           # required improvement (the original used 1e-4 with metric = "loss")
    },
    "evaluation": {
        "shuffle": False,           # fixed per-patch shuffle at validation and test (S2)
        "val_patches_per_leaf": 256,        # fixed subset per validation leaf; 0 = all
        "test_patches_per_leaf": 0,         # 0 = every interior patch (as in all runs so far)
    },
}
CHOICES = {("data", "scaler"): ("train", "train+val"),
           ("training", "objective"): ("patch", "leaf"),
           ("early_stopping", "metric"): ("balanced_accuracy", "loss")}
SMOKE = {("folds",): [0], ("early_stopping", "max_epochs"): 2, ("early_stopping", "patience"): 1,
         ("training", "samples_per_class"): 2_000, ("evaluation", "val_patches_per_leaf"): 32}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def set_value(cfg: dict, path: tuple, value, where: str):
    """Put value at path, checking the key exists and the type matches the default.

    Validation happens here, per setting, so a typo (unknown key) or a wrong type (lr = "fast")
    stops the program before any data is loaded, naming the file or --set that caused it."""
    node, default = cfg, DEFAULTS
    for key in path[:-1]:
        if key not in default or not isinstance(default[key], dict):
            raise SystemExit(f"{where}: unknown section '{'.'.join(path[:-1])}'")
        node, default = node[key], default[key]
    key, name = path[-1], ".".join(path)
    if key not in default:
        raise SystemExit(f"{where}: unknown setting '{name}' (see configs/template.toml)")
    ref = default[key]
    if isinstance(ref, dict):
        raise SystemExit(f"{where}: '{name}' is a section, set its keys instead")
    # TOML reads lr = 1 as an integer; accept it where the default is a float
    if isinstance(ref, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    # bool is a subclass of int in Python, so it is excluded explicitly from the number checks
    if isinstance(ref, bool):
        ok = isinstance(value, bool)
    elif isinstance(ref, list):
        ok = isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    else:
        ok = isinstance(value, type(ref)) and not isinstance(value, bool)
    if not ok:
        raise SystemExit(f"{where}: '{name}' must be {type(ref).__name__}"
                         + (" of integers" if isinstance(ref, list) else "") + f", got {value!r}")
    node[key] = value


def flatten(d: dict, prefix=()) -> list:
    """Nested dict -> [(key path tuple, value), ...]."""
    out = []
    for k, v in d.items():
        out += flatten(v, prefix + (k,)) if isinstance(v, dict) else [(prefix + (k,), v)]
    return out


def parse_set(text: str):
    """'section.key=value' -> (key path, value); the value is read as TOML, otherwise as a plain string."""
    key, sep, raw = text.partition("=")
    if not sep:
        raise SystemExit(f"--set {text}: expected section.key=value")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw                                     # a bare word, e.g. name=my_run
    return tuple(key.strip().split(".")), value


def load_config(path: Path | None, overrides: list[str], smoke: bool) -> dict:
    """DEFAULTS, then the config file, then the --set overrides, then the smoke settings; validated."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "rb") as f:
            for key, value in flatten(tomllib.load(f)):
                set_value(cfg, key, value, str(path))
    for text in overrides:
        key, value = parse_set(text)
        set_value(cfg, key, value, f"--set {text}")
    if smoke:
        for key, value in SMOKE.items():
            set_value(cfg, key, value, "--smoke")
    cfg["name"] = cfg["name"].strip() or (path.stem if path else "default")
    check_config(cfg)
    return cfg


def check_config(cfg: dict):
    """Checks the type check cannot do (allowed values, ranges, leaf batch size); exits listing every error.

    All errors are collected first so one run of the command shows everything to fix."""
    bad = []
    for (sec, key), allowed in CHOICES.items():
        if cfg[sec][key] not in allowed:
            bad.append(f"{sec}.{key} must be one of {allowed}, got {cfg[sec][key]!r}")
    # the name becomes a folder name; 'smoke' is reserved for runs/custom/smoke/
    if not re.fullmatch(r"[A-Za-z0-9_.+ -]+", cfg["name"]) or cfg["name"] == "smoke":
        bad.append(f"name {cfg['name']!r}: use letters, digits, spaces and _ . + - only (and not 'smoke')")
    if not cfg["folds"] or len(set(cfg["folds"])) != len(cfg["folds"]) or not set(cfg["folds"]) <= set(range(N_FOLDS)):
        bad.append(f"folds must be distinct values from 0-{N_FOLDS - 1}, got {cfg['folds']}")
    for sec, key in [("model", "channels"), ("training", "batch_size"), ("training", "samples_per_class"),
                     ("training", "leaves_per_class"), ("training", "patches_per_leaf"),
                     ("early_stopping", "patience"), ("early_stopping", "max_epochs")]:
        if cfg[sec][key] < 1:
            bad.append(f"{sec}.{key} must be at least 1")
    for key in ("p_kernels", "i_kernels", "d_kernels"):
        if not cfg["model"][key] or min(cfg["model"][key]) < 1:
            bad.append(f"model.{key} must be a non-empty list of positive kernel sizes")
    if cfg["training"]["lr"] <= 0:
        bad.append("training.lr must be positive")
    tr = cfg["training"]
    # the leaf objective builds each batch from whole leaves, so the batch must split into them exactly
    if tr["objective"] == "leaf" and len(CLASSES) * tr["leaves_per_class"] * tr["patches_per_leaf"] != tr["batch_size"]:
        bad.append(f"objective = 'leaf': 4 x leaves_per_class x patches_per_leaf must equal batch_size "
                   f"(4 x {tr['leaves_per_class']} x {tr['patches_per_leaf']} != {tr['batch_size']})")
    if len(CLASSES) * tr["samples_per_class"] < tr["batch_size"]:
        bad.append("training.samples_per_class is too small for one batch")
    if bad:
        raise SystemExit("config errors:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------
# Pieces of controlled.py (copied so this script depends only on 11_pmmfnet.py)
# --------------------------------------------------------------------------
def split_fold(meta, fold):
    """Training leaves of a test fold -> (fit, validation = row (fold + 1) mod 5, validation fold).

    Validation is a whole row, like the test split, so neighbouring vines cannot leak into it."""
    train_idx, _ = fold_indices(meta, fold)
    val_fold = (fold + 1) % N_FOLDS
    is_val = meta.fold.to_numpy()[train_idx] == val_fold
    return train_idx[~is_val], train_idx[is_val], val_fold


def eval_permutations(n_patches: int, device) -> torch.Tensor:
    """(n_patches, 24) uint8: a fixed random order of the 24 neighbours for every interior patch.

    Drawn once with a seeded torch generator in 1M-patch chunks (part of the random stream, keep
    it) and cached as .npy, so this script and controlled.py use the same permutations."""
    if not EVAL_PERM_PATH.exists():
        EVAL_PERM_PATH.parent.mkdir(exist_ok=True)
        g = torch.Generator().manual_seed(EVAL_PERM_SEED)    # own generator: does not touch the global seeds
        perms = np.empty((n_patches, 24), np.uint8)          # NumPy only as the .npy cache format
        for s in range(0, n_patches, 1_000_000):
            e = min(s + 1_000_000, n_patches)
            perms[s:e] = torch.rand(e - s, 24, generator=g).argsort(1).numpy().astype(np.uint8)
        np.save(EVAL_PERM_PATH, perms)
    perms = np.load(EVAL_PERM_PATH)
    assert perms.shape == (n_patches, 24)
    return torch.from_numpy(perms).to(device)


def make_eval_input(pix, rows, lo, hi, perm=None) -> torch.Tensor:
    """Evaluation patches in the original order (perm None) or the fixed neighbour order perm.

    rows: (B, 25) pixel indices; perm: (B, 24); returns (B, 1, 25, 208), channels-last."""
    if perm is None:
        return m11.make_input(pix, rows, lo, hi, shuffle=False)
    x = (pix[rows.long()] - lo) / (hi - lo)
    others = m11.NEIGHBOURS.to(x.device)
    x[:, others] = torch.gather(x, 1, others[perm.long()][:, :, None].expand(-1, -1, x.shape[2]))
    x = F.pad(x, (0, m11.N_BANDS_IN - x.shape[2])).unsqueeze(1)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def leaf_probs(model, pix, nbr, ids, owner, n_leaves, lo, hi, perms=None, chunk=2048) -> np.ndarray:
    """Mean softmax over the given patches of each leaf -> (n_leaves, 4).

    ids: patch ids to score; owner: for each patch, the position of its leaf (0..n_leaves-1)."""
    if perms is None:
        return m11.leaf_probs(model, pix, nbr, ids, owner, n_leaves, lo, hi, chunk)
    model.eval()
    sums = torch.zeros(n_leaves, len(CLASSES), device=pix.device)
    for s in range(0, len(ids), chunk):
        sel = ids[s:s + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(make_eval_input(pix, nbr[sel], lo, hi, perms[sel]))
        sums.index_add_(0, owner[s:s + chunk], torch.softmax(logits.float(), 1))
    return (sums / torch.bincount(owner, minlength=n_leaves)[:, None]).cpu().numpy()   # NumPy for sklearn / files


def leaf_mil_loss(logits: torch.Tensor, y_leaf: torch.Tensor, per_leaf: int) -> torch.Tensor:
    """Cross-entropy of the leaf probability = mean softmax over the leaf's per_leaf consecutive patches.

    The leaf probability is what is evaluated at test time; the clamp keeps log() finite."""
    prob = torch.softmax(logits, 1).view(-1, per_leaf, logits.shape[1]).mean(1)
    return F.nll_loss(torch.log(prob.clamp_min(1e-12)), y_leaf)


def class_sequence(leaves: torch.Tensor, n: int, device) -> torch.Tensor:
    """n leaves of one class: shuffled passes over the class's leaves, concatenated (even use of leaves)."""
    reps = -(-n // len(leaves))                              # ceil(n / number of leaves)
    return torch.cat([leaves[torch.randperm(len(leaves), device=device)] for _ in range(reps)])[:n]


def leaf_ranges(pix: torch.Tensor, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-leaf min and max over all of the leaf's pixels and bands -> two (n_leaves,) float64 arrays.

    Computed on the GPU and copied once (min / max are exact, so the values equal a per-leaf
    .item() loop); returned as NumPy because they are indexed with the NumPy fold indices."""
    lo = torch.stack([pix[s:s + n].min() for s, n in zip(meta.px_start, meta.n_px)])
    hi = torch.stack([pix[s:s + n].max() for s, n in zip(meta.px_start, meta.n_px)])
    return lo.cpu().numpy().astype(np.float64), hi.cpu().numpy().astype(np.float64)


@contextmanager
def branch_kernels(model_cfg):
    """PMMFNet reads its branch kernel lists from module constants; set them while building.

    pmmfnet.py is checksum-protected, so the constants are swapped here (and always restored)
    instead of adding constructor arguments to the network."""
    saved = pmmfnet.P_KERNELS, pmmfnet.I_KERNELS, pmmfnet.D_KERNELS
    pmmfnet.P_KERNELS, pmmfnet.I_KERNELS, pmmfnet.D_KERNELS = (
        tuple(model_cfg["p_kernels"]), tuple(model_cfg["i_kernels"]), tuple(model_cfg["d_kernels"]))
    try:
        yield
    finally:
        pmmfnet.P_KERNELS, pmmfnet.I_KERNELS, pmmfnet.D_KERNELS = saved


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_fold(model, data, fit_idx, val_idx, lo, hi, cfg, device, tag,
               log: logging.Logger) -> tuple[dict, dict]:
    """controlled.py's training loop with every setting taken from cfg.

    data = (pix, nbr, p_start, p_count, cls, perms); fit_idx / val_idx: leaf indices; lo, hi: the
    scaler. One line per epoch goes to `log`. Returns (record with history, best state_dict); the
    model ends with the best epoch's weights."""
    pix, nbr, p_start, p_count, cls, perms = data
    tr, es, ev = cfg["training"], cfg["early_stopping"], cfg["evaluation"]
    model.to(device, memory_format=torch.channels_last)     # faster convs under bf16
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    y_t = torch.from_numpy(cls).to(device)
    by_class = [torch.from_numpy(fit_idx[cls[fit_idx] == c]).to(device) for c in range(len(CLASSES))]
    assert all(len(b) for b in by_class), "a class has no training leaf"
    assert not set(fit_idx.tolist()) & set(val_idx.tolist()), "validation leaf in training"
    ps, pc = (torch.from_numpy(a).to(device) for a in (p_start, p_count))
    # Fixed validation patches (own seed): every epoch is scored on the same inputs, so
    # epoch-to-epoch changes come from the model only.
    val_ids, val_owner = m11.patch_ids(val_idx, p_start, p_count, device,
                                       per_leaf=ev["val_patches_per_leaf"] or None, seed=VAL_PATCH_SEED)
    eval_perm = perms if ev["shuffle"] else None
    y_val = cls[val_idx]
    bs, by_bacc = tr["batch_size"], es["metric"] == "balanced_accuracy"
    best = {"val_balanced_accuracy": -np.inf, "val_loss": np.inf, "epoch": 0}
    best_state, wait, history, t0 = None, 0, [], time.time()

    for epoch in range(1, es["max_epochs"] + 1):
        model.train()
        running = torch.zeros((), device=device)                    # summed on the GPU: no sync per step
        if tr["objective"] == "patch":
            # 1a. Class-balanced patch sampling (the original loop): epoch_leaves lists each class's
            # training leaves K/M times; every entry gets a fresh random interior centre.
            lf = m11.epoch_leaves(by_class, tr["samples_per_class"], device)
            centre = ps[lf] + (torch.rand(len(lf), device=device) * pc[lf]).long()
            steps = len(lf) // bs                                       # the last partial batch is dropped
            for b in range(steps):
                sl = slice(b * bs, (b + 1) * bs)
                x = m11.make_input(pix, nbr[centre[sl]], lo, hi, shuffle=tr["shuffle"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y_t[lf[sl]])    # loss in float32, not bf16
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.detach()
        else:                                                           # leaf-level MIL
            # 1b. Nested sampling: leaves first (per-class shuffled sequences for the whole epoch,
            # k per class per step, so batches are balanced), then per_leaf random interior
            # patches of each leaf, kept consecutive because leaf_mil_loss averages them in groups.
            k, per_leaf = tr["leaves_per_class"], tr["patches_per_leaf"]
            steps = tr["samples_per_class"] * len(CLASSES) // bs
            seqs = [class_sequence(b, k * steps, device) for b in by_class]
            for st in range(steps):
                leaves = torch.cat([s[k * st:k * (st + 1)] for s in seqs])
                rep = leaves.repeat_interleave(per_leaf)
                centre = ps[rep] + (torch.rand(len(rep), device=device) * pc[rep]).long()
                x = m11.make_input(pix, nbr[centre], lo, hi, shuffle=tr["shuffle"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                loss = leaf_mil_loss(logits.float(), y_t[leaves], per_leaf)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.detach()

        # 2. Leaf-level validation: the same leaf probabilities as at test time.
        P = leaf_probs(model, pix, nbr, val_ids, val_owner, len(val_idx), lo, hi, eval_perm)
        val_loss = m11.balanced_ce(P, y_val)
        val_bacc = balanced_accuracy_score(y_val, P.argmax(1))
        if not np.isfinite(val_loss):
            raise RuntimeError(f"{tag}: non-finite validation loss at epoch {epoch}")
        # 3. Checkpoint rule. "balanced_accuracy": the reported metric picks the checkpoint; it moves
        # in steps, so exact ties are common and the lower validation loss breaks them. A tie-break
        # saves the weights but does not reset patience (only a real rise buys more epochs); 1e-12
        # absorbs float noise. "loss": the original pipeline's rule.
        if by_bacc:
            improved = val_bacc > best["val_balanced_accuracy"] + max(es["min_delta"], 1e-12)
            tie_better = abs(val_bacc - best["val_balanced_accuracy"]) <= 1e-12 and val_loss < best["val_loss"]
        else:
            improved, tie_better = val_loss < best["val_loss"] - es["min_delta"], False
        if improved or tie_better:
            best = {"val_balanced_accuracy": val_bacc, "val_loss": val_loss, "epoch": epoch}
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        wait = 0 if improved else wait + 1
        history.append({"epoch": epoch, "train_loss": running.item() / steps, "val_loss": val_loss,
                        "val_balanced_accuracy": val_bacc, "seconds": round(time.time() - t0, 1)})
        flag = "*best*" if improved else ("*best (tie, lower loss)*" if tie_better else f"(no gain {wait}/{es['patience']})")
        log.info(f"   [{tag}] epoch {epoch:3d}  train loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  "
                 f"val bal.acc {val_bacc:.3f}  {flag}  ({time.time() - t0:.0f}s)")
        if wait >= es["patience"]:
            break
    train_seconds = time.time() - t0

    # 4. Restore the best epoch and re-score it, to confirm the saved checkpoint is the recorded
    # one (non-deterministic GPU kernels can cause small differences: logged, not fatal).
    model.load_state_dict(best_state)
    P = leaf_probs(model, pix, nbr, val_ids, val_owner, len(val_idx), lo, hi, eval_perm)
    check = {"val_loss": m11.balanced_ce(P, y_val), "val_balanced_accuracy": balanced_accuracy_score(y_val, P.argmax(1))}
    check["matches"] = bool(abs(check["val_loss"] - best["val_loss"]) < 1e-3
                            and abs(check["val_balanced_accuracy"] - best["val_balanced_accuracy"]) < 1e-9)
    if not check["matches"]:
        log.warning(f"   [{tag}] WARNING restored weights give val loss {check['val_loss']:.4f} / bal.acc "
                    f"{check['val_balanced_accuracy']:.3f}, recorded {best['val_loss']:.4f} / "
                    f"{best['val_balanced_accuracy']:.3f} (GPU non-determinism)")
    log.info(f"   [{tag}] restored epoch {best['epoch']} (val bal.acc {best['val_balanced_accuracy']:.3f}, "
             f"val loss {best['val_loss']:.4f}); stopped after epoch {epoch}")
    record = {"best_epoch": best["epoch"], "stopped_epoch": epoch,
              "best_val_balanced_accuracy": best["val_balanced_accuracy"], "val_loss_at_best": best["val_loss"],
              "restore_check": check, "optimizer_steps_per_epoch": steps, "patch_inputs_per_epoch": steps * bs,
              "train_seconds": round(train_seconds, 1), "history": history}
    return record, best_state


def run_fold(cfg, d, fold, device, log) -> tuple[dict, dict, np.ndarray]:
    """Seed, train and test one fold -> (record, best state_dict, (n_test, 4) test leaf probabilities).

    d: the dict from load_data. The order of the steps fixes the random streams (seeding, model
    initialisation, training draws, test patch subset); keep it unchanged."""
    name, meta, cls = cfg["name"], d["meta"], d["cls"]
    perms = d["perms"] if cfg["evaluation"]["shuffle"] else None
    data = (d["pix"], d["nbr"], d["p_start"], d["p_count"], cls, d["perms"])
    _, test_idx = fold_indices(meta, fold)
    seed_everything(cfg["seed"] + fold)
    fit_idx, val_idx, val_fold = split_fold(meta, fold)
    # scaler: global min / max over the training leaves ('train' keeps held-out pixels out of it)
    idx = np.concatenate([fit_idx, val_idx]) if cfg["data"]["scaler"] == "train+val" else fit_idx
    lo, hi = float(d["leaf_lo"][idx].min()), float(d["leaf_hi"][idx].max())
    log.info(f"fold {fold}: {len(fit_idx)} train / {len(val_idx)} validation (fold {val_fold}) / "
             f"{len(test_idx)} test leaves; scaler on {cfg['data']['scaler']} ({len(idx)} leaves) [{lo:.4f}, {hi:.4f}]")
    with branch_kernels(cfg["model"]):
        model = pmmfnet.PMMFNet(n_out=len(CLASSES), channels=cfg["model"]["channels"], n_bands=m11.N_BANDS_IN)
    record, state = train_fold(model, data, fit_idx, val_idx, lo, hi, cfg, device, f"{name} f{fold}", log)
    t0 = time.time()
    n_test = cfg["evaluation"]["test_patches_per_leaf"] or None
    ids, owner = m11.patch_ids(test_idx, d["p_start"], d["p_count"], device, per_leaf=n_test, seed=TEST_PATCH_SEED)
    P_test = leaf_probs(model, d["pix"], d["nbr"], ids, owner, len(test_idx), lo, hi, perms)
    record |= {"test_fold": fold, "validation_fold": val_fold, "n_train_leaves": len(fit_idx),
               "n_validation_leaves": len(val_idx), "n_test_leaves": len(test_idx),
               "test_patches": int(len(ids)), "test_seconds": round(time.time() - t0, 1),
               "scaler": {"fitted_on": cfg["data"]["scaler"], "n_leaves": int(len(idx)), "min": lo, "max": hi},
               "seeds": {"torch_numpy_python": cfg["seed"] + fold, "validation_patch_subset": VAL_PATCH_SEED,
                         "eval_permutations": EVAL_PERM_SEED if perms is not None else None},
               "test_balanced_accuracy": balanced_accuracy_score(cls[test_idx], P_test.argmax(1))}
    return record, state, P_test


# --------------------------------------------------------------------------
# Data, output folder and logging
# --------------------------------------------------------------------------
_DATA = {}


def load_data(device, need_perms: bool):
    """Pixels on the GPU and the patch table, loaded once per process (shared by all configs).

    Returns a dict: meta, cls (n_leaves,), pix (n_pixels, 204) float32 GPU, nbr (n_patches, 25)
    GPU, p_start / p_count (first patch and count per leaf), leaf_lo / leaf_hi (per-leaf min /
    max), perms (S2 evaluation permutations, loaded when first needed)."""
    if not _DATA:
        meta = load_meta()
        nbr_np, p_start, p_count = m11.patch_table(m11.build_pixel_index(meta), meta)
        pix = m11.pixels_to_gpu(device)
        leaf_lo, leaf_hi = leaf_ranges(pix, meta)
        _DATA.update(meta=meta, cls=leaf_classes(meta), nbr=torch.from_numpy(nbr_np).to(device),
                     p_start=p_start, p_count=p_count, pix=pix, perms=None, leaf_lo=leaf_lo, leaf_hi=leaf_hi)
    if need_perms and _DATA["perms"] is None:
        _DATA["perms"] = eval_permutations(len(_DATA["nbr"]), device)
    return _DATA


def fold_assignment(meta, cls) -> pd.DataFrame:
    """One row per leaf with its role (train / validation / test) when each fold is the test fold."""
    roles = pd.DataFrame({"path": meta.path, "variety": meta.variety, "row": meta.row, "fold": meta.fold,
                          "class": [CLASSES[c] for c in cls]})
    for fold in range(N_FOLDS):
        fit_idx, val_idx, _ = split_fold(meta, fold)
        role = np.full(len(meta), "test", dtype=object)
        role[fit_idx], role[val_idx] = "train", "validation"
        roles[f"role_when_test_fold_{fold}"] = role
    return roles


def prepare_output(cfg: dict, smoke: bool, overwrite: bool, log: logging.Logger) -> Path | None:
    """Create runs/custom/<name>/ (smoke: runs/custom/smoke/<name>/). Returns None when the run is
    already complete; refuses a different config under an existing name."""
    out = RUNS / ("smoke" if smoke else "") / cfg["name"]
    saved = out / "config.json"
    if out.exists() and (smoke or overwrite):
        shutil.rmtree(out)                                  # smoke runs always start clean
    # Resume safety: finished folds are reused only if they were trained with exactly these
    # settings. The description is free text, so editing it does not block a resume.
    elif saved.exists() and ({**json.loads(saved.read_text()), "description": ""} != {**cfg, "description": ""}):
        raise SystemExit(f"{out} holds a run with different settings (see its config.json); "
                         "use another name (--set name=...) or --overwrite to replace it")
    elif (out / f"{cfg['name']}_metrics.json").exists():   # the report is written last: run complete
        log.info(f"{cfg['name']}: already complete ({out}), skipping")
        return None
    (out / "folds").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    saved.write_text(json.dumps(cfg, indent=2))
    return out


class _LinesToLog(io.TextIOBase):
    """Writable text stream that sends every complete line written to it to log.info."""

    def __init__(self, log: logging.Logger):
        super().__init__()
        self.log, self.buf = log, ""

    def write(self, text):
        self.buf += text
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            self.log.info(line)
        return len(text)


@contextmanager
def prints_to_log(log: logging.Logger):
    """Route print() output of helpers this script does not own (11_pmmfnet.py's data loading,
    four_class.save_results' "[saved]" line) through `log`, so it reaches log.txt as before.

    The logger's terminal handler holds the real stdout (created in main before any redirect),
    so the redirected lines are not sent back into this stream."""
    sink = _LinesToLog(log)
    with redirect_stdout(sink):
        yield
    if sink.buf:
        log.info(sink.buf)


def log_traceback_to_file(log: logging.Logger):
    """Append the current exception's traceback to the logger's files only (Python itself prints
    it on the terminal)."""
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            h.stream.write(traceback.format_exc())
            h.flush()


# --------------------------------------------------------------------------
# Running one config
# --------------------------------------------------------------------------
def run(cfg: dict, smoke: bool, overwrite: bool, log: logging.Logger):
    """Run one config; while it runs, every message is also appended to the run's log.txt."""
    out = prepare_output(cfg, smoke, overwrite, log)
    if out is None:
        return
    get_logger(log.name, log_file=out / "log.txt")         # append mode: a resumed run continues its log
    try:
        _run(cfg, out, smoke, log)
    except BaseException:
        log_traceback_to_file(log)
        raise
    finally:
        close_log_file(log)                                 # the next config gets its own log.txt


def _run(cfg: dict, out: Path, smoke: bool, log: logging.Logger):
    """Train and evaluate every requested fold (finished folds are reloaded) and write the report."""
    name, t_start, device = cfg["name"], time.time(), torch.device("cuda")
    # deterministic = false keeps the original's cudnn.benchmark (fast; same-seed reruns differ slightly)
    det = cfg["deterministic"]
    torch.backends.cudnn.benchmark = not det
    torch.backends.cudnn.deterministic = det
    torch.use_deterministic_algorithms(det, warn_only=True)
    log.info(f"=== {name} ({time.strftime('%Y-%m-%d %H:%M:%S')}){' [smoke]' if smoke else ''}: {cfg['description']}")
    log.info("settings: " + ", ".join(f"{'.'.join(k)}={v}" for k, v in flatten(cfg) if k[0] not in ("name", "description")))
    with prints_to_log(log):
        d = load_data(device, cfg["evaluation"]["shuffle"])
    meta = d["meta"]
    fold_assignment(meta, d["cls"]).to_csv(out / "fold_assignment.csv", index=False)

    P = np.full((len(meta), len(CLASSES)), np.nan)     # out-of-fold leaf probabilities; NaN = not predicted
    records = {}
    for fold in cfg["folds"]:
        rec_path, p_path = out / "folds" / f"fold{fold}.json", out / "folds" / f"fold{fold}.npz"
        _, test_idx = fold_indices(meta, fold)
        # Resume: both files are written only after a fold has finished, so if both exist the
        # fold is complete (prepare_output has checked the config is the same) and is reloaded.
        if rec_path.exists() and p_path.exists():
            records[fold] = json.loads(rec_path.read_text())
            P[test_idx] = np.load(p_path)["P"]
            log.info(f"fold {fold}: loaded from {rec_path}")
            continue
        record, state, P[test_idx] = run_fold(cfg, d, fold, device, log)
        torch.save(state, out / "checkpoints" / f"fold{fold}.pt")
        np.savez(p_path, P=P[test_idx], test_idx=test_idx)
        rec_path.write_text(json.dumps(record, indent=2))
        records[fold] = record
        log.info(f"[{name}] fold {fold}: test 4-class bal.acc {record['test_balanced_accuracy']:.3f} "
                 f"(best epoch {record['best_epoch']}, {record['train_seconds'] / 60:.1f} min training)")
        del state
        torch.cuda.empty_cache()

    runtime = time.time() - t_start
    with prints_to_log(log):
        md = write_report(cfg, meta, P, records, runtime, out)
    oof = pd.read_csv(out / f"{name}_oof.csv")
    assert len(oof) == sum(len(fold_indices(meta, k)[1]) for k in cfg["folds"]) and oof.path.is_unique
    if not smoke:
        log.info("\n" + md)
    log.info(f"{name}: total {runtime / 60:.1f} min -> {out}")


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def write_report(cfg: dict, meta, P, records: dict, runtime: float, out: Path) -> str:
    """Write <name>_oof.csv and <name>_metrics.md/.json (via four_class.save_results); return the markdown."""
    name = cfg["name"]
    fold_md = ["", "## Per fold", "",
               "| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |",
               "|---|---|---|---|---|---|---|---|---|"]
    for fold, r in records.items():
        fold_md.append(f"| {fold} | {r['validation_fold']} | [{r['scaler']['min']:.4f}, {r['scaler']['max']:.4f}] | "
                       f"{r['best_epoch']} | {r['stopped_epoch']} | {fmt(r['best_val_balanced_accuracy'])} | "
                       f"{r['val_loss_at_best']:.4f} | {fmt(r['test_balanced_accuracy'])} | {r['train_seconds'] / 60:.1f} |")
    settings = {".".join(k): v for k, v in flatten(cfg) if k[0] not in ("name", "description")}
    title = f"PMMFNet custom run {name}" + (f" - {cfg['description']}" if cfg["description"] else "")
    return save_results(name, title, meta, P, settings, extra_md="\n".join(fold_md) + "\n",
                        extra_json={"folds": records, "runtime_seconds": round(runtime, 1),
                                    "description": cfg["description"], "resolved_config": cfg},
                        pred=P.argmax(1), folder=out, extended=True)   # final leaf prediction = most probable class


def summary_rows() -> list[dict]:
    """One row per custom, smoke and controlled run that has a metrics file."""
    rows = []
    sources = [("custom", p) for p in sorted(RUNS.glob("*/*_metrics.json"))]
    sources += [("smoke", p) for p in sorted((RUNS / "smoke").glob("*/*_metrics.json"))]
    sources += [("controlled", HERE / "runs" / e / f"{e}_metrics.json") for e in ("A1", "A2", "S1", "S2", "S3", "L2")]
    for kind, path in sources:
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        mc, folds = m["metrics"]["four_class"], m["folds"]
        rows.append({"run": path.parent.name, "kind": kind, "description": m.get("description", ""),
                     "folds": ",".join(sorted(folds)),
                     "val bal. acc.": np.mean([f["best_val_balanced_accuracy"] for f in folds.values()]),
                     "test bal. acc.": mc["balanced_accuracy"], "macro-F1": mc["macro_f1"],
                     "mean OvR AUC": mc["auc_ovr_mean"]["pooled"],
                     "within-variety AUC": mc["auc_ovr_mean"]["within_variety"],
                     "RB within": m["metrics"]["per_disease"]["red_blotch"]["auc_within_variety"],
                     "LR within": m["metrics"]["per_disease"]["leafroll3"]["auc_within_variety"],
                     "mean best epoch": np.mean([f["best_epoch"] for f in folds.values()]),
                     "runtime min": m["runtime_seconds"] / 60})
    return rows


def summary_markdown(rows: list[dict]) -> str:
    """The summary table, one section per kind of run."""
    cols = [c for c in rows[0] if c not in ("kind", "description")]
    L = ["# PMMFNet runs - summary", "",
         "val bal. acc. = mean best validation 4-class balanced accuracy (choose configurations by this); "
         "the other metrics are on the held-out test leaves of the listed folds (runs on different folds "
         "are not comparable). Same-seed reruns differed by up to 0.038 test bal. acc. per fold (REPORT.md).", ""]
    for kind in ("custom", "smoke", "controlled"):
        part = [r for r in rows if r["kind"] == kind]
        if not part:
            continue
        L += [f"## {kind}", "", "| " + " | ".join(cols) + " | description |", "|---" * (len(cols) + 1) + "|"]
        for r in part:
            L.append("| " + " | ".join(f"{r[c]:.1f}" if c in ("runtime min", "mean best epoch") else
                                       fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols)
                     + f" | {r['description']} |")
        L.append("")
    return "\n".join(L)


def summary(log: logging.Logger):
    """runs/custom/summary.md: one row per custom, smoke and controlled run."""
    rows = summary_rows()
    if not rows:
        log.info("no runs yet")
        return
    text = summary_markdown(rows)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "summary.md").write_text(text)
    log.info(text + f"\n[saved] {RUNS / 'summary.md'}")


def main():
    """Parse the command line, validate every config before starting, then run them one after another."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path, help="TOML config files, run one after another")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a setting of every config (repeatable)")
    ap.add_argument("--smoke", action="store_true", help="tiny run of fold 0 (2 epochs) to check a config")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run of the same name first")
    ap.add_argument("--summary", action="store_true", help="write runs/custom/summary.md (after any runs)")
    args = ap.parse_args()
    if not args.configs and not args.summary:
        ap.error("give at least one config file (start from configs/template.toml), or --summary")
    cfgs = [load_config(p, args.overrides, args.smoke) for p in args.configs]   # check every config first
    names = [c["name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit(f"two configs share a name: {names}")
    # Terminal handler now, before any stdout redirect; each run adds (and removes) its log.txt.
    log = get_logger("pmmfnet.run_experiment")
    for cfg in cfgs:
        run(cfg, args.smoke, args.overwrite, log)
    if args.summary:
        summary(log)


if __name__ == "__main__":
    main()
