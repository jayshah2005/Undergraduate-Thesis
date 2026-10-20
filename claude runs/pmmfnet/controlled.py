"""Controlled PMMFNet experiments (Classification/plan.md): A1, A2, S1, S2, S3, L2.

Run from claude runs/pmmfnet/:
    ../../.venv/bin/python -u controlled.py --experiment A1            # 5 folds -> runs/A1/
    ../../.venv/bin/python -u controlled.py --experiment A1 --smoke    # tiny run + checks -> runs/smoke/A1/
    ./run_all.sh                                                       # smoke tests, all experiments, comparison

Reads the shared data cache (../shared/cache/, via common.py and 11_pmmfnet.py) and, for S2, the
fixed evaluation permutations in cache/. Writes runs/<exp>/ (smoke: runs/smoke/<exp>/):
<exp>_oof.csv, <exp>_metrics.md/.json, folds/fold<k>.json/.npz (written as each fold finishes, so a
crashed run resumes), checkpoints/fold<k>.pt, fold_assignment.csv. Progress goes to stdout through
the shared logger (../shared/logs.py); run_all.sh redirects it into runs/<exp>/log.txt.

Terms: fold k = test row k of every variety (5 row folds); the validation row of fold k is row
(k+1) mod 5; bal. acc. = 4-class balanced accuracy (mean recall over the classes) of the leaf
predictions; leaf probability = mean softmax over the leaf's patches; MIL = multiple-instance
learning (a loss on the leaf probability instead of on each patch).

Everything not listed here is the original pipeline (11_pmmfnet.py), whose helpers are imported
unchanged: the same 5 row folds and validation row ((k+1) mod 5), native-resolution interior
5x5 patches, 204 bands zero-filled to 208 after scaling, PMMFNet (32 channels), Adam at
lr 1e-5 (constant), batch 256 patch inputs, 100,000 patches per class per epoch, bf16,
max 50 epochs, patience 5, the same fixed validation patches (up to 256 per leaf, seed 1),
leaf probability = mean softmax over the leaf's patches (all interior patches at test time),
seeds: torch / numpy / python seeded with the fold number.

WHAT CHANGES (all experiments)
  Checkpoint selection: highest 4-class leaf-level validation balanced accuracy; on an exact
  tie the lower validation loss (leaf-level class-balanced cross-entropy, the original's
  criterion) wins. Patience counts epochs without a rise in balanced accuracy (a tie-break
  update saves the checkpoint but does not reset patience). The best weights are restored
  and saved to runs/<exp>/checkpoints/fold<k>.pt.

THE EXPERIMENTS                scaler fitted on       train shuffle   val/test shuffle      loss
  A1  checkpoint rule only     train + val leaves     random          none                  patch
  A2  + training-only scaler   train leaves           random          none                  patch
  S1  no shuffle               train leaves           none            none                  patch
  S2  shuffle everywhere       train leaves           random          fixed per patch       patch
  S3  shuffle in training      train leaves           random          none                  patch
  L2  leaf-level MIL loss      train leaves           random          none                  leaf
  S3 has exactly A2's settings (the plan defines both that way); it is rerun with the same
  seeds, so A2 vs S3 measures run-to-run noise (GPU non-determinism). L1, the plan's
  patch-level reference for L2, is also A2's configuration and is reported as the A2 run.

  Scaler: one global min and max over all bands and pixels of the scaler's leaves; the same
  values transform training, validation and test patches; no clipping (as in the original).
  S2: one permutation of the 24 neighbours per interior patch, drawn once from a fixed seed
  and cached (cache/eval_permutations_seed12345.npy), so a patch is shuffled identically in
  every validation epoch and at test time. Spectral bands are never reordered.
  L2: each optimizer step takes 8 training leaves (2 per class, drawn from per-class
  sequences of shuffled leaves, i.e. with replacement across the epoch), 32 random interior
  patches per leaf (256 inputs), averages the 32 softmax outputs per leaf and applies the
  cross-entropy of that leaf probability vs the leaf's class. No class weights (the batches
  are balanced). 1562 steps per epoch = 399,872 patch inputs, the same as the patch-level
  runs (1562 x 256).
"""
import argparse
import importlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))       # common.py, torch_common.py, four_class.py, logs.py
from common import N_FOLDS, fold_indices, load_meta                          # noqa: E402
from four_class import CLASSES, fmt, leaf_classes, save_results              # noqa: E402
from logs import get_logger                                                  # noqa: E402
from pmmfnet import PMMFNet                                                  # noqa: E402
from torch_common import seed_everything                                     # noqa: E402

m11 = importlib.import_module("11_pmmfnet")                # the original pipeline's helpers

EXPERIMENTS = {
    "A1": {"scaler": "train+val", "train_shuffle": True, "eval_shuffle": False, "objective": "patch",
           "what": "original configuration, balanced-accuracy checkpoint selection"},
    "A2": {"scaler": "train", "train_shuffle": True, "eval_shuffle": False, "objective": "patch",
           "what": "A1 + scaler fitted on training leaves only"},
    "S1": {"scaler": "train", "train_shuffle": False, "eval_shuffle": False, "objective": "patch",
           "what": "A2 without spatial shuffle anywhere"},
    "S2": {"scaler": "train", "train_shuffle": True, "eval_shuffle": True, "objective": "patch",
           "what": "A2 + fixed spatial shuffle of validation and test patches"},
    "S3": {"scaler": "train", "train_shuffle": True, "eval_shuffle": False, "objective": "patch",
           "what": "shuffle in training only (= A2 settings, rerun)"},
    "L2": {"scaler": "train", "train_shuffle": True, "eval_shuffle": False, "objective": "leaf",
           "what": "A2 with a leaf-level (multiple-instance) training loss"},
}
EVAL_PERM_SEED = 12345
VAL_PATCH_SEED = 1                          # the original's fixed validation patch subset
LEAVES_PER_CLASS_PER_BATCH, PATCHES_PER_LEAF = 2, 32
ORIGINAL_SCALER = {0: (0.0077, 1.6282), 1: (0.0073, 2.8844), 2: (0.0073, 2.8844),
                   3: (0.0073, 2.8844), 4: (0.0073, 2.8844)}     # from runs/original/logs/11.md


# --------------------------------------------------------------------------
# Inputs: the original make_input, plus a deterministic shuffle for evaluation (S2)
# --------------------------------------------------------------------------
def eval_permutations(n_patches: int, device) -> torch.Tensor:
    """(n_patches, 24) uint8: a fixed random order of the 24 neighbours for every interior patch.

    Drawn once with a seeded torch generator, in chunks of 1M patches (the chunking is part of
    the random stream, so it must not change), and cached as .npy so every run and
    run_experiment.py use the same permutations."""
    path = HERE / "cache" / f"eval_permutations_seed{EVAL_PERM_SEED}.npy"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        g = torch.Generator().manual_seed(EVAL_PERM_SEED)    # own generator: does not touch the global seeds
        perms = np.empty((n_patches, 24), np.uint8)          # NumPy only as the .npy cache format
        for s in range(0, n_patches, 1_000_000):
            e = min(s + 1_000_000, n_patches)
            perms[s:e] = torch.rand(e - s, 24, generator=g).argsort(1).numpy().astype(np.uint8)
        np.save(path, perms)
    perms = np.load(path)
    assert perms.shape == (n_patches, 24)
    return torch.from_numpy(perms).to(device)


def make_eval_input(pix, rows, lo, hi, perm=None) -> torch.Tensor:
    """Evaluation patches: original order (perm None, = the original make_input) or the fixed
    per-patch neighbour order `perm` (B, 24); the centre row 12 never moves.

    pix: (n_pixels, 204) float32 on the GPU; rows: (B, 25) pixel indices of each patch;
    lo, hi: the scaler's global min / max. Returns (B, 1, 25, 208), channels-last."""
    if perm is None:
        return m11.make_input(pix, rows, lo, hi, shuffle=False)
    x = (pix[rows.long()] - lo) / (hi - lo)
    others = m11.NEIGHBOURS.to(x.device)
    src = others[perm.long()]
    x[:, others] = torch.gather(x, 1, src[:, :, None].expand(-1, -1, x.shape[2]))
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
        x = make_eval_input(pix, nbr[sel], lo, hi, perms[sel])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        sums.index_add_(0, owner[s:s + chunk], torch.softmax(logits.float(), 1))
    return (sums / torch.bincount(owner, minlength=n_leaves)[:, None]).cpu().numpy()   # NumPy for sklearn / files


# --------------------------------------------------------------------------
# Training objectives and scaler
# --------------------------------------------------------------------------
def leaf_mil_loss(logits: torch.Tensor, y_leaf: torch.Tensor, per_leaf: int) -> torch.Tensor:
    """Cross-entropy of the leaf probability = mean softmax over the leaf's `per_leaf`
    consecutive patches (logits: (n_leaves * per_leaf, 4), y_leaf: (n_leaves,)).

    The leaf probability is the same quantity used to predict a leaf at test time, so this loss
    trains what is evaluated; the clamp keeps log() finite if a class probability underflows."""
    prob = torch.softmax(logits, 1).view(-1, per_leaf, logits.shape[1]).mean(1)
    return F.nll_loss(torch.log(prob.clamp_min(1e-12)), y_leaf)


def class_sequence(leaves: torch.Tensor, n: int, device) -> torch.Tensor:
    """n leaves of one class: shuffled passes over the class's leaves, concatenated.

    Each pass visits every leaf once before any repeats, so leaves are used evenly."""
    reps = -(-n // len(leaves))                              # ceil(n / number of leaves)
    return torch.cat([leaves[torch.randperm(len(leaves), device=device)] for _ in range(reps)])[:n]


def leaf_ranges(pix: torch.Tensor, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-leaf min and max over all of the leaf's pixels and bands -> two (n_leaves,) float64 arrays.

    Computed on the GPU and copied once (min / max are exact, so the values equal a per-leaf
    .item() loop); returned as NumPy because they are indexed with the NumPy fold indices."""
    lo = torch.stack([pix[s:s + n].min() for s, n in zip(meta.px_start, meta.n_px)])
    hi = torch.stack([pix[s:s + n].max() for s, n in zip(meta.px_start, meta.n_px)])
    return lo.cpu().numpy().astype(np.float64), hi.cpu().numpy().astype(np.float64)


def scaler_range(cfg, fit_idx, val_idx, leaf_lo, leaf_hi):
    """Global min / max over the leaves the scaler is fitted on (training leaves, plus the
    validation leaves for 'train+val') -> (lo, hi, those leaves' indices).

    'train' keeps validation and test pixels out of the scaler, so nothing about held-out
    leaves reaches the inputs; 'train+val' reproduces the original pipeline."""
    idx = np.concatenate([fit_idx, val_idx]) if cfg["scaler"] == "train+val" else fit_idx
    return float(leaf_lo[idx].min()), float(leaf_hi[idx].max()), idx


# --------------------------------------------------------------------------
# Training one fold
# --------------------------------------------------------------------------
def train_fold(model, data, fit_idx, val_idx, lo, hi, cfg, args, device, tag,
               log: logging.Logger) -> tuple[dict, dict]:
    """Train with the balanced-accuracy checkpoint rule; returns (record, best state_dict).

    data = (pix, nbr, p_start, p_count, cls, perms) from load_data; fit_idx / val_idx: leaf
    indices; lo, hi: the scaler. One line per epoch goes to `log`; the model ends with the best
    epoch's weights. The record holds the best / stopping epoch, the restore check and the
    per-epoch history."""
    pix, nbr, p_start, p_count, cls, perms = data
    model.to(device, memory_format=torch.channels_last)     # faster convs under bf16
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_t = torch.from_numpy(cls).to(device)
    by_class = [torch.from_numpy(fit_idx[cls[fit_idx] == c]).to(device) for c in range(len(CLASSES))]
    assert all(len(b) for b in by_class), "a class has no training leaf"
    assert not set(fit_idx.tolist()) & set(val_idx.tolist()), "validation leaf in training"
    ps, pc = (torch.from_numpy(a).to(device) for a in (p_start, p_count))
    # The validation patches are a fixed subset (own seed), so every epoch is scored on the same
    # inputs and epoch-to-epoch changes come from the model only.
    val_ids, val_owner = m11.patch_ids(val_idx, p_start, p_count, device, per_leaf=args.val_patches,
                                       seed=VAL_PATCH_SEED)
    eval_perm = perms if cfg["eval_shuffle"] else None
    y_val = cls[val_idx]
    bs = args.batch_size
    best = {"val_balanced_accuracy": -np.inf, "val_loss": np.inf, "epoch": 0}
    best_state, wait, history, t0 = None, 0, [], time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = torch.zeros((), device=device)                # summed on the GPU: no sync per step
        if cfg["objective"] == "patch":                          # the original loop
            # 1a. Class-balanced patch sampling: epoch_leaves lists each class's training leaves
            # K/M times (K per class, shuffled together); every entry then gets a fresh random
            # interior centre, so each epoch sees new patches of every leaf.
            lf = m11.epoch_leaves(by_class, args.samples_per_class, device)
            centre = ps[lf] + (torch.rand(len(lf), device=device) * pc[lf]).long()
            steps = len(lf) // bs                                # the last partial batch is dropped
            for b in range(steps):
                sl = slice(b * bs, (b + 1) * bs)
                x = m11.make_input(pix, nbr[centre[sl]], lo, hi, shuffle=cfg["train_shuffle"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y_t[lf[sl]])   # loss in float32, not bf16
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.detach()
        else:                                                    # leaf-level MIL
            # 1b. Nested sampling: leaves first, then patches inside each leaf. The per-class leaf
            # sequences for the whole epoch are drawn up front; step st takes the next k leaves of
            # every class (8 leaves, balanced, so no class weights) and 32 random interior
            # patches of each, kept consecutive per leaf because leaf_mil_loss averages them in
            # groups of PATCHES_PER_LEAF. Same number of steps as the patch objective.
            steps = args.samples_per_class * len(CLASSES) // bs
            k = LEAVES_PER_CLASS_PER_BATCH
            seqs = [class_sequence(b, k * steps, device) for b in by_class]
            for st in range(steps):
                leaves = torch.cat([s[k * st:k * (st + 1)] for s in seqs])               # 8 leaves
                rep = leaves.repeat_interleave(PATCHES_PER_LEAF)                          # 256 inputs
                centre = ps[rep] + (torch.rand(len(rep), device=device) * pc[rep]).long()
                x = m11.make_input(pix, nbr[centre], lo, hi, shuffle=cfg["train_shuffle"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                loss = leaf_mil_loss(logits.float(), y_t[leaves], PATCHES_PER_LEAF)
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
        # 3. Checkpoint rule. Balanced accuracy is the reported metric, so it picks the checkpoint
        # (the original used the loss). On 200 validation leaves it moves in steps, so exact ties
        # are common: the lower validation loss breaks them. A tie-break saves the weights but
        # does not reset patience - only a real rise buys more epochs. 1e-12 absorbs float noise.
        improved = val_bacc > best["val_balanced_accuracy"] + 1e-12
        tie_better = abs(val_bacc - best["val_balanced_accuracy"]) <= 1e-12 and val_loss < best["val_loss"]
        if improved or tie_better:
            best = {"val_balanced_accuracy": val_bacc, "val_loss": val_loss, "epoch": epoch}
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        wait = 0 if improved else wait + 1
        history.append({"epoch": epoch, "train_loss": running.item() / steps, "val_loss": val_loss,
                        "val_balanced_accuracy": val_bacc, "seconds": round(time.time() - t0, 1)})
        flag = "*best*" if improved else ("*best (tie, lower loss)*" if tie_better else f"(no gain {wait}/{args.patience})")
        log.info(f"   [{tag}] epoch {epoch:3d}  train loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  "
                 f"val bal.acc {val_bacc:.3f}  {flag}  ({time.time() - t0:.0f}s)")
        if wait >= args.patience:
            break
    train_seconds = time.time() - t0

    # 4. Restore the best epoch and re-score it: confirms the saved checkpoint is the recorded one
    # (small differences can come from non-deterministic GPU kernels; they are logged, not fatal).
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


def run_fold(exp, cfg, args, meta, data, leaf_lo, leaf_hi, fold, device, log) -> tuple[dict, dict, np.ndarray]:
    """Seed, train and test one fold -> (record, best state_dict, (n_test, 4) test leaf probabilities).

    The order of the steps below fixes the random streams (seeding, model initialisation,
    then training draws); keep it unchanged so runs stay comparable."""
    pix, nbr, p_start, p_count, cls, perms = data
    _, test_idx = fold_indices(meta, fold)
    seed_everything(fold)
    fit_idx, val_idx, val_fold = split_fold(meta, fold)
    lo, hi, scaler_idx = scaler_range(cfg, fit_idx, val_idx, leaf_lo, leaf_hi)
    log.info(f"fold {fold}: {len(fit_idx)} train / {len(val_idx)} validation (fold {val_fold}) / "
             f"{len(test_idx)} test leaves; scaler on {cfg['scaler']} ({len(scaler_idx)} leaves) [{lo:.4f}, {hi:.4f}]")
    model = PMMFNet(n_out=len(CLASSES), channels=args.channels, n_bands=m11.N_BANDS_IN)
    record, state = train_fold(model, data, fit_idx, val_idx, lo, hi, cfg, args, device, f"{exp} f{fold}", log)
    t0 = time.time()
    ids, owner = m11.patch_ids(test_idx, p_start, p_count, device)          # every interior patch
    P_test = leaf_probs(model, pix, nbr, ids, owner, len(test_idx), lo, hi, perms if cfg["eval_shuffle"] else None)
    record |= {"test_fold": fold, "validation_fold": val_fold, "n_train_leaves": len(fit_idx),
               "n_validation_leaves": len(val_idx), "n_test_leaves": len(test_idx),
               "test_patches": int(len(ids)), "test_seconds": round(time.time() - t0, 1),
               "scaler": {"fitted_on": cfg["scaler"], "n_leaves": int(len(scaler_idx)), "min": lo, "max": hi},
               "seeds": {"torch_numpy_python": fold, "validation_patch_subset": VAL_PATCH_SEED,
                         "eval_permutations": EVAL_PERM_SEED if cfg["eval_shuffle"] else None},
               "test_balanced_accuracy": balanced_accuracy_score(cls[test_idx], P_test.argmax(1))}
    return record, state, P_test


# --------------------------------------------------------------------------
# Smoke checks (run with --smoke before the full experiments)
# --------------------------------------------------------------------------
def smoke_checks(exp, cfg, meta, cls, data, leaf_lo, leaf_hi, device, log):
    """Checks run with --smoke: fold splits, no scaler leakage, patch shapes and zero-fill, centre
    pixel never moved, fixed eval permutations (S2), leaf-level loss and batch composition (L2).
    Raises AssertionError on the first failure; reports each passed group as a [smoke] line."""
    pix, nbr, p_start, p_count, _, perms = data
    for fold in range(N_FOLDS):
        train_idx, test_idx = fold_indices(meta, fold)
        fit_idx, val_idx, val_fold = split_fold(meta, fold)
        assert np.array_equal(np.sort(np.concatenate([fit_idx, val_idx])), np.sort(train_idx))
        assert len(set(fit_idx) | set(val_idx) | set(test_idx)) == len(meta)
        assert (meta.fold.to_numpy()[val_idx] == val_fold).all() and (meta.fold.to_numpy()[test_idx] == fold).all()
        lo, hi, used = scaler_range(cfg, fit_idx, val_idx, leaf_lo, leaf_hi)
        if cfg["scaler"] == "train+val":
            assert (round(lo, 4), round(hi, 4)) == ORIGINAL_SCALER[fold], (fold, lo, hi)
        else:
            assert not set(used.tolist()) & set(val_idx.tolist()) and not set(used.tolist()) & set(test_idx.tolist())
            assert lo == leaf_lo[fit_idx].min() and hi == leaf_hi[fit_idx].max()
        log.info(f"  [smoke] fold {fold}: {len(fit_idx)}/{len(val_idx)}/{len(test_idx)} leaves, "
                 f"scaler ({cfg['scaler']}) [{lo:.4f}, {hi:.4f}]")
    rows = torch.arange(64, device=device)
    x = make_eval_input(pix, nbr[rows], 0.0, 1.0)
    assert x.shape == (64, 1, 25, m11.N_BANDS_IN) and (x[..., 204:] == 0).all()
    assert torch.equal(x, m11.make_input(pix, nbr[rows], 0.0, 1.0, shuffle=False))
    xs = m11.make_input(pix, nbr[rows], 0.0, 1.0, shuffle=True)
    assert torch.equal(xs[:, :, 12], x[:, :, 12]), "training shuffle moved the centre pixel"
    if cfg["eval_shuffle"]:
        p = perms[rows].long()
        assert torch.equal(p.sort(1).values, torch.arange(24, device=device).expand(64, -1)), "not a permutation"
        x1, x2 = make_eval_input(pix, nbr[rows], 0.0, 1.0, perms[rows]), make_eval_input(pix, nbr[rows], 0.0, 1.0, perms[rows])
        assert torch.equal(x1, x2) and torch.equal(x1[:, :, 12], x[:, :, 12]) and not torch.equal(x1, x)
        assert torch.equal(x1.sort(2).values, x.sort(2).values), "eval shuffle changed pixel values"
    if cfg["objective"] == "leaf":
        logits = torch.randn(8 * PATCHES_PER_LEAF, 4, device=device)
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
        # reference value computed leaf by leaf, independently of leaf_mil_loss's reshape
        manual = torch.stack([-torch.log(torch.softmax(logits[i * 32:(i + 1) * 32], 1).mean(0)[y[i]])
                              for i in range(8)]).mean().item()
        assert abs(leaf_mil_loss(logits, y, PATCHES_PER_LEAF).item() - manual) < 1e-5
        fit_idx = split_fold(meta, 0)[0]
        by_class = [torch.from_numpy(fit_idx[cls[fit_idx] == c]).to(device) for c in range(4)]
        seqs = [class_sequence(b, 2 * 10, device) for b in by_class]
        leaves = torch.cat([s[0:2] for s in seqs])
        assert (torch.from_numpy(cls).to(device)[leaves] == y).all() and len(leaves.repeat_interleave(32)) == 256
    log.info(f"  [smoke] {exp}: splits, scaler, patch shapes, centre pixel"
             + (", fixed eval permutations" if cfg["eval_shuffle"] else "")
             + (", leaf-level loss and batch composition" if cfg["objective"] == "leaf" else "") + " OK")


def check_smoke_outputs(exp, meta, out, log):
    """After a smoke run: one output row per fold-0 test leaf, checkpoint and fold record written."""
    oof = pd.read_csv(out / f"{exp}_oof.csv")
    assert len(oof) == len(fold_indices(meta, 0)[1]) and oof.path.is_unique, "one row per test leaf expected"
    assert (out / "checkpoints" / "fold0.pt").exists() and (out / "folds" / "fold0.json").exists()
    log.info(f"  [smoke] {exp}: outputs OK ({len(oof)} leaf rows) in {out}")


# --------------------------------------------------------------------------
# Data and folds
# --------------------------------------------------------------------------
def split_fold(meta, fold):
    """Training leaves of a test fold -> (fit leaves, validation leaves = row (fold + 1) mod 5, validation fold).

    Validation is a whole row, like the test split, so neighbouring vines of a training row
    cannot leak into it."""
    train_idx, _ = fold_indices(meta, fold)
    val_fold = (fold + 1) % N_FOLDS
    is_val = meta.fold.to_numpy()[train_idx] == val_fold
    return train_idx[~is_val], train_idx[is_val], val_fold


def load_data(cfg, device):
    """Leaf table, classes, patch table and all pixels on the GPU -> (meta, cls, data, leaf_lo, leaf_hi).

    data = (pix (n_pixels, 204) float32 GPU, nbr (n_patches, 25) pixel indices GPU, p_start,
    p_count (first patch and patch count per leaf), cls (n_leaves,) class, perms (S2 only))."""
    meta = load_meta()
    cls = leaf_classes(meta)
    nbr_np, p_start, p_count = m11.patch_table(m11.build_pixel_index(meta), meta)
    nbr = torch.from_numpy(nbr_np).to(device)
    pix = m11.pixels_to_gpu(device)
    leaf_lo, leaf_hi = leaf_ranges(pix, meta)
    perms = eval_permutations(len(nbr_np), device) if cfg["eval_shuffle"] else None
    return meta, cls, (pix, nbr, p_start, p_count, cls, perms), leaf_lo, leaf_hi


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


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(exp, cfg, args, folds, meta, P, records, runtime, out) -> str:
    """Write <exp>_oof.csv and <exp>_metrics.md/.json (via four_class.save_results); return the markdown."""
    settings = {"experiment": exp, **{k: v for k, v in cfg.items() if k != "what"}, "lr": args.lr,
                "batch": args.batch_size, "K per class": args.samples_per_class, "max epochs": args.max_epochs,
                "patience": args.patience, "val patches per leaf": args.val_patches, "channels": args.channels,
                "bands": f"204 zero-filled to {m11.N_BANDS_IN}", "checkpoint": "max val bal.acc, ties -> min val loss",
                "folds": ",".join(map(str, folds))}
    fold_md = ["", "## Per fold", "",
               "| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |",
               "|---|---|---|---|---|---|---|---|---|"]
    for fold, r in records.items():
        fold_md.append(f"| {fold} | {r['validation_fold']} | [{r['scaler']['min']:.4f}, {r['scaler']['max']:.4f}] | "
                       f"{r['best_epoch']} | {r['stopped_epoch']} | {fmt(r['best_val_balanced_accuracy'])} | "
                       f"{r['val_loss_at_best']:.4f} | {fmt(r['test_balanced_accuracy'])} | {r['train_seconds'] / 60:.1f} |")
    return save_results(exp, f"PMMFNet controlled experiment {exp} - {cfg['what']}", meta, P, settings,
                        extra_md="\n".join(fold_md) + "\n",
                        extra_json={"folds": records, "runtime_seconds": round(runtime, 1), "description": cfg["what"]},
                        pred=P.argmax(1), folder=out, extended=True)   # final leaf prediction = most probable class


# --------------------------------------------------------------------------
def parse_args():
    """Command line; --smoke shrinks the run to fold 0, 2 epochs and a few patches."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--samples-per-class", type=int, default=100_000)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-patches", type=int, default=256)
    ap.add_argument("--channels", type=int, default=32)
    args = ap.parse_args()
    if args.smoke:
        args.max_epochs, args.patience, args.samples_per_class, args.val_patches = 2, 1, 2_000, 32
    return args


def main():
    """Run one experiment over the 5 folds (finished folds are reloaded) and write its outputs."""
    args = parse_args()
    # Terminal only: run_all.sh redirects stdout into runs/<exp>/log.txt, so a log file here
    # would write every line twice.
    log = get_logger("pmmfnet.controlled")
    exp, cfg = args.experiment, EXPERIMENTS[args.experiment]
    folds = [0] if args.smoke else list(range(N_FOLDS))
    out = HERE / "runs" / ("smoke" if args.smoke else "") / exp
    if args.smoke and out.exists():
        shutil.rmtree(out)                                  # smoke tests always start clean
    (out / "folds").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    if (out / f"{exp}_metrics.json").exists() and not args.smoke:
        log.info(f"{exp}: already complete ({out}), skipping")
        return

    t_start = time.time()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True               # as the original: fast, but reruns are not bit-identical
    meta, cls, data, leaf_lo, leaf_hi = load_data(cfg, device)
    log.info(f"{exp}: {cfg['what']}  {cfg}")
    if args.smoke:
        smoke_checks(exp, cfg, meta, cls, data, leaf_lo, leaf_hi, device, log)
    fold_assignment(meta, cls).to_csv(out / "fold_assignment.csv", index=False)

    P = np.full((len(meta), len(CLASSES)), np.nan)     # out-of-fold leaf probabilities; NaN = not predicted
    records = {}
    for fold in folds:
        rec_path, p_path = out / "folds" / f"fold{fold}.json", out / "folds" / f"fold{fold}.npz"
        _, test_idx = fold_indices(meta, fold)
        # Resume: both files are written only after a fold has finished, so if both exist the
        # fold is complete and is reloaded instead of retrained.
        if rec_path.exists() and p_path.exists():
            records[fold] = json.loads(rec_path.read_text())
            P[test_idx] = np.load(p_path)["P"]
            log.info(f"fold {fold}: loaded from {rec_path}")
            continue
        record, state, P[test_idx] = run_fold(exp, cfg, args, meta, data, leaf_lo, leaf_hi, fold, device, log)
        torch.save(state, out / "checkpoints" / f"fold{fold}.pt")
        np.savez(p_path, P=P[test_idx], test_idx=test_idx)
        rec_path.write_text(json.dumps(record, indent=2))
        records[fold] = record
        log.info(f"[{exp}] fold {fold}: test 4-class bal.acc {record['test_balanced_accuracy']:.3f} "
                 f"(best epoch {record['best_epoch']}, {record['train_seconds'] / 60:.1f} min training)")
        del state
        torch.cuda.empty_cache()

    if not args.smoke:
        assert np.isfinite(P).all(), "some leaves have no prediction"
    runtime = time.time() - t_start
    md = write_report(exp, cfg, args, folds, meta, P, records, runtime, out)
    if args.smoke:
        check_smoke_outputs(exp, meta, out, log)
    else:
        assert len(pd.read_csv(out / f"{exp}_oof.csv")) == len(meta)
        log.info("\n" + md)
    log.info(f"{exp}: total {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
