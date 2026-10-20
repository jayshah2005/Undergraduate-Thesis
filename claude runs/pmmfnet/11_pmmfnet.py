"""APPROACH 11 - PMMFNet (Wang et al., Scientific Reports 15:29288, 2025) on 5x5 pixel patches.

Run:   python experiments/11_pmmfnet.py           (5 folds, early stopping; ~50 s per epoch on GPU)
       python experiments/11_pmmfnet.py --quick   (fold 0, 3 tiny epochs, patience 1; writes results/quick/)

The network is in pmmfnet.py, with the figure-by-figure reading of the paper and
the text-vs-figure decisions (the text is followed). This script is the pipeline
around it: patches, spatial shuffle, class-balanced sampling, training with
early stopping, leaf scores, and the metrics (confusion matrix, balanced
accuracy, ROC AUC).

CLASSES  The paper is multi-class (softmax + cross-entropy). The two labels become
  4 mutually exclusive classes, class = red_blotch + 2 * leafroll3:
      0 healthy    1 red blotch only    2 leafroll only    3 both
  so P(red blotch) = P(1) + P(3), P(leafroll) = P(2) + P(3), P(healthy) = P(0).

WHAT THE PAPER DOES -> WHAT THIS SCRIPT DOES
  Sample      a pixel + its 5x5 neighbourhood, flattened to a 25 x B image
              -> same, cut at NATIVE resolution from the original cubes (the
                 128 px images in cache/ are downscaled, so they are not used).
                 Centres are interior leaf pixels (whole 5x5 window on the
                 leaf); the paper's scenes have no zero background to fall into.
  Bands       "all bands less than 200 are filled with zeros ... 25 x 200"
              -> all 204 bands kept and zero-filled to 208, the next multiple of
                 8, so the network's three 1x2 pools divide evenly (map 25 x 26).
  Scaling     "values ... normalized to the range of 0-1"
              -> one min and one max over all bands and pixels of the non-test
                 leaves (training + validation rows); zero-fill after scaling.
  Shuffle     keep the centre pixel, randomly permute the other 24 (Fig. 1)
              -> a fresh permutation every time a training patch is drawn;
                 validation and test patches are not shuffled.
  Samples     K = 100,000 per class; each of a class's M samples is shuffled K/M times
              -> every epoch draws K = 100,000 patches per class (400,000): each
                 training leaf of the class K/M times, a random interior centre each time.
  Loss        cross-entropy, Adam, learning rate 0.00001, batch 256
              -> same; constant learning rate, no weight decay, no class weights
                 (the sampling balances the classes, as in the paper).
  Train/test  200 labelled pixels per class; the rest is the test set
              -> 5 folds = held-out rows (common.py). Of the 4 training rows, one
                 (row of fold k+1) is the validation row for early stopping;
                 row-grouped like the test split, so neighbouring vines cannot
                 leak into it. The model trains on the other 3 rows.
  Epochs      not given -> early stopping [not in the paper]: after every epoch,
                 leaf-level class-balanced cross-entropy on the validation row
                 (a fixed set of up to 256 patches per leaf); stop after
                 `patience` epochs without improvement and restore the weights
                 of the best epoch.
  Output      per-pixel class -> leaf probabilities = mean softmax over all of
              the leaf's interior pixels.
  bf16 mixed precision and channels-last memory layout on the GPU for speed (not in the
  paper; torch.compile was tried but gave non-finite gradients together with bf16).

METRICS (results/11_pmmfnet_metrics.md and .json; per-leaf probabilities in 11_pmmfnet_oof.csv)
  * 4 classes: confusion matrix of the argmax class, balanced accuracy (mean
    recall over the classes), one-vs-rest ROC AUC per class and their mean.
  * Per disease (red blotch; leafroll without Cab Franc, which has 1 positive;
    any disease = not healthy): 2x2 confusion matrix and balanced accuracy from
    the argmax class, ROC AUC from the summed class probabilities.
  * ROC AUC pooled over all leaves and within-variety (leaves compared only with
    leaves of the same variety and fold - the honest number, see common.py).
  * The same per variety, and the early-stopping record of every fold.
"""
import argparse
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import balanced_accuracy_score

from common import (CACHE_DIR, H5_PATH, N_FOLDS, fold_indices, load_meta, load_pixels, n_folds,
                    print_fold_summary)
from four_class import CLASSES, fmt, leaf_classes, save_results
from pmmfnet import PMMFNet
from torch_common import get_device, seed_everything

PATCH = 5
CENTRE = PATCH * PATCH // 2                 # row 12 of the 25 = the centre pixel
N_BANDS_IN = 208                            # 204 bands zero-filled to a multiple of 8
INDEX_PATH = CACHE_DIR / "pmmf_pixel_index.npy"
# common.H5_PATH predates experiments/ moving into "claude runs/"; fall back one level up.
H5 = H5_PATH if H5_PATH.exists() else H5_PATH.parent.parent / H5_PATH.name

warnings.filterwarnings("ignore", message="Using padding='same' with even kernel")


# --------------------------------------------------------------------------
# 1. Where is each pixel? A native-resolution map from (row, col) to its row in pixels.npy
# --------------------------------------------------------------------------
def build_pixel_index(meta: pd.DataFrame) -> np.ndarray:
    """(N_leaves, max_H, max_W) int32: index into pixels.npy, -1 = background.

    pixels.npy (00_build_cache.py) holds cube_t[mask] in row-major (line, sample)
    order, with NaN -> 0 and mask = "any band != 0". The mask is rebuilt here
    exactly that way from the h5 (read once, ~30 s, then cached) and checked
    against pixels.npy: if the map were off, every patch would be scrambled.
    """
    if INDEX_PATH.exists():
        return np.load(INDEX_PATH)
    t0 = time.time()
    pixels = load_pixels(mmap=True)
    rng = np.random.default_rng(0)
    index = np.full((len(meta), meta.height.max(), meta.width.max()), -1, dtype=np.int32)
    with h5py.File(H5, "r") as f:
        for i, leaf in enumerate(meta.itertuples()):
            cube_t = np.nan_to_num(f[leaf.path]["cube"][:].transpose(0, 2, 1), nan=0.0)
            mask = (cube_t != 0).any(axis=2)
            assert mask.sum() == leaf.n_px, f"{leaf.path}: {mask.sum()} leaf pixels, cache has {leaf.n_px}"
            rows, cols = np.nonzero(mask)                                    # row-major, like cube_t[mask]
            index[i, rows, cols] = leaf.px_start + np.arange(leaf.n_px)
            for k in rng.choice(leaf.n_px, size=5, replace=False):          # spot-check 5 pixels per leaf
                assert np.array_equal(pixels[index[i, rows[k], cols[k]]], cube_t[rows[k], cols[k]]), leaf.path
            if (i + 1) % 200 == 0:
                print(f"  pixel index: {i + 1}/{len(meta)} leaves ({time.time() - t0:.0f}s)")
    np.save(INDEX_PATH, index)
    print(f"[saved] {INDEX_PATH}  (checked: pixel counts of all leaves, 5 pixels per leaf)")
    return index


def patch_table(index: np.ndarray, meta: pd.DataFrame):
    """All interior 5x5 patches as rows of 25 pixel indices (row-major, centre = column 12).

    Returns nbr (n_patches, 25) int32, and per leaf the first patch and the count.
    """
    nbr, start, count, s = [], [], [], 0
    for i, leaf in enumerate(meta.itertuples()):
        win = sliding_window_view(index[i, :leaf.height, :leaf.width], (PATCH, PATCH))
        ok = (win >= 0).all(axis=(2, 3))                                     # whole window on the leaf
        nbr.append(win[ok].reshape(-1, PATCH * PATCH))
        start.append(s)
        count.append(int(ok.sum()))
        s += count[-1]
    assert min(count) > 0, "a leaf has no interior 5x5 patch"
    return np.concatenate(nbr), np.array(start), np.array(count)


def pixels_to_gpu(device, chunk=500_000) -> torch.Tensor:
    """All leaf pixels, all 204 bands, as one float32 tensor on the GPU (~6.3 GB)."""
    px = load_pixels(mmap=True)
    out = torch.empty(px.shape, dtype=torch.float32, device=device)
    for s in range(0, px.shape[0], chunk):
        out[s:s + chunk] = torch.from_numpy(np.ascontiguousarray(px[s:s + chunk])).to(device)
    return out


def patch_ids(leaf_idx, p_start, p_count, device, per_leaf=None, seed=0):
    """Patch ids of the given leaves, and for each patch the position of its leaf in leaf_idx.

    per_leaf=None: every interior patch; otherwise a fixed random subset of up to per_leaf.
    """
    rng = np.random.default_rng(seed)
    ids, owner = [], []
    for k, i in enumerate(leaf_idx):
        a = np.arange(p_start[i], p_start[i] + p_count[i])
        if per_leaf is not None and len(a) > per_leaf:
            a = np.sort(rng.choice(a, per_leaf, replace=False))
        ids.append(a)
        owner.append(np.full(len(a), k))
    return (torch.from_numpy(np.concatenate(ids)).to(device),
            torch.from_numpy(np.concatenate(owner)).to(device))


# --------------------------------------------------------------------------
# 2. Patches -> network input, with the paper's spatial shuffle
# --------------------------------------------------------------------------
NEIGHBOURS = torch.tensor([j for j in range(PATCH * PATCH) if j != CENTRE])


def make_input(pix, nbr_rows, lo, hi, shuffle: bool) -> torch.Tensor:
    """(B, 25) pixel indices -> (B, 1, 25, 208) min-max scaled, zero-filled patches.

    shuffle=True: the 24 neighbours of every patch are put in a random order
    (its own permutation per patch); the centre stays at row 12 (paper Fig. 1).
    """
    x = (pix[nbr_rows.long()] - lo) / (hi - lo)                              # (B, 25, 204), 0-1 on train leaves
    if shuffle:
        others = NEIGHBOURS.to(x.device)
        order = torch.rand(len(x), len(others), device=x.device).argsort(1)
        src = others[order]                                                  # (B, 24) shuffled source rows
        x[:, others] = torch.gather(x, 1, src[:, :, None].expand(-1, -1, x.shape[2]))
    x = F.pad(x, (0, N_BANDS_IN - x.shape[2])).unsqueeze(1)                  # zeros after the last band
    return x.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------
# 3. Training with early stopping, and leaf-level prediction
# --------------------------------------------------------------------------
def epoch_leaves(by_class, k: int, device) -> torch.Tensor:
    """One epoch of training leaves, K per class: each of a class's M leaves K // M
    times, plus K % M of them once more at random (paper: K/M shuffles per sample)."""
    out = []
    for leaves in by_class:
        m = len(leaves)
        out += [leaves.repeat(k // m), leaves[torch.randperm(m, device=device)[:k % m]]]
    lf = torch.cat(out)
    return lf[torch.randperm(len(lf), device=device)]


@torch.no_grad()
def leaf_probs(model, pix, nbr, ids, owner, n_leaves, lo, hi, chunk=2048) -> np.ndarray:
    """Mean softmax over the given patches of each leaf -> (n_leaves, 4)."""
    model.eval()
    sums = torch.zeros(n_leaves, len(CLASSES), device=pix.device)
    for s in range(0, len(ids), chunk):
        x = make_input(pix, nbr[ids[s:s + chunk]], lo, hi, shuffle=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=pix.is_cuda):
            logits = model(x)
        sums.index_add_(0, owner[s:s + chunk], torch.softmax(logits.float(), 1))
    return (sums / torch.bincount(owner, minlength=n_leaves)[:, None]).cpu().numpy()


def balanced_ce(P: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy of leaf probabilities, averaged within each class, then over classes."""
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1.0))
    return float(np.mean([ll[y == c].mean() for c in np.unique(y)]))


def train_fold(model, pix, nbr, p_start, p_count, fit_idx, val_idx, cls, lo, hi,
               args, device, tag) -> dict:
    """Adam at a constant learning rate; each epoch = fresh centres + fresh shuffles.
    Returns the early-stopping record; the model ends with the best epoch's weights."""
    model.to(device, memory_format=torch.channels_last)                      # faster convs under bf16
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_t = torch.from_numpy(cls).to(device)
    by_class = [torch.from_numpy(fit_idx[cls[fit_idx] == c]).to(device) for c in range(len(CLASSES))]
    assert all(len(b) for b in by_class), "a class has no training leaf"
    assert not set(torch.cat(by_class).tolist()) & set(val_idx.tolist()), "validation leaf in training"
    ps, pc = (torch.from_numpy(a).to(device) for a in (p_start, p_count))
    val_ids, val_owner = patch_ids(val_idx, p_start, p_count, device, per_leaf=args.val_patches, seed=1)
    y_val = cls[val_idx]
    bs = args.batch_size
    best, best_epoch, best_state, wait, history, t0 = np.inf, 0, None, 0, [], time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        lf = epoch_leaves(by_class, args.samples_per_class, device)
        centre = ps[lf] + (torch.rand(len(lf), device=device) * pc[lf]).long()
        running, n_batches = torch.zeros((), device=device), len(lf) // bs   # drop the last partial batch
        for b in range(n_batches):
            sl = slice(b * bs, (b + 1) * bs)
            x = make_input(pix, nbr[centre[sl]], lo, hi, shuffle=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y_t[lf[sl]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.detach()

        P = leaf_probs(model, pix, nbr, val_ids, val_owner, len(val_idx), lo, hi)
        val_loss = balanced_ce(P, y_val)
        val_bacc = balanced_accuracy_score(y_val, P.argmax(1))
        if not np.isfinite(val_loss):
            raise RuntimeError(f"{tag}: non-finite validation loss at epoch {epoch}")
        if val_loss < best - args.min_delta:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        history.append({"epoch": epoch, "train_loss": running.item() / n_batches,
                        "val_loss": val_loss, "val_balanced_accuracy": val_bacc})
        print(f"   [{tag}] epoch {epoch:3d}  train loss {history[-1]['train_loss']:.4f}  "
              f"val loss {val_loss:.4f}  val bal.acc {val_bacc:.3f}  "
              f"{'*best*' if wait == 0 else f'(no gain {wait}/{args.patience})'}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if wait >= args.patience:
            break

    model.load_state_dict(best_state)
    check = balanced_ce(leaf_probs(model, pix, nbr, val_ids, val_owner, len(val_idx), lo, hi), y_val)
    assert abs(check - best) < 1e-3, f"restored weights give val loss {check:.4f}, best was {best:.4f}"
    print(f"   [{tag}] restored epoch {best_epoch} (val loss {check:.4f}); stopped after epoch {epoch}")
    return {"best_epoch": best_epoch, "stopped_epoch": epoch, "best_val_loss": best,
            "val_balanced_accuracy_at_best": history[best_epoch - 1]["val_balanced_accuracy"],
            "history": history}


# --------------------------------------------------------------------------
# 4. Report: the 4-class metrics live in four_class.py; this adds the early-stopping table
# --------------------------------------------------------------------------
def stopping_markdown(stopping: dict) -> str:
    rows = [f"| {fold} | {s['val_fold']} | {s['best_epoch']} | {s['stopped_epoch']} | "
            f"{s['best_val_loss']:.4f} | {fmt(s['val_balanced_accuracy_at_best'])} |"
            for fold, s in stopping.items()]
    return "\n".join(["", "## Early stopping", "",
                      "Validation = one training row per fold (not used for training); loss = leaf-level "
                      "class-balanced cross-entropy. Full curves in the .json file.", "",
                      "| test fold | validation fold | best epoch | stopped after | best val loss | val bal. acc. at best |",
                      "|---|---|---|---|---|---|"] + rows) + "\n"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)                        # paper
    ap.add_argument("--batch-size", type=int, default=256)                   # paper
    ap.add_argument("--samples-per-class", type=int, default=100_000)        # paper: K per class
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--val-patches", type=int, default=256)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--folds", default=None, help="comma-separated, e.g. 0,1 (default: all)")
    ap.add_argument("--name", default="11_pmmfnet")
    args = ap.parse_args()
    if args.quick:
        args.max_epochs, args.patience, args.samples_per_class, args.val_patches = 3, 1, 2_000, 32
    folds = [0] if args.quick else ([int(k) for k in args.folds.split(",")] if args.folds
                                    else range(n_folds()))

    device = get_device()
    if device.type != "cuda":
        raise SystemExit("No CUDA GPU visible to PyTorch; this pipeline is meant to run on the GPU.")
    torch.backends.cudnn.benchmark = True
    meta = load_meta()
    cls = leaf_classes(meta)
    index = build_pixel_index(meta)
    nbr_np, p_start, p_count = patch_table(index, meta)
    nbr = torch.from_numpy(nbr_np).to(device)
    pix = pixels_to_gpu(device)
    # per-leaf min / max, so each fold's 0-1 scaling uses its non-test leaves only
    leaf_lo = np.array([pix[s:s + n].min().item() for s, n in zip(meta.px_start, meta.n_px)])
    leaf_hi = np.array([pix[s:s + n].max().item() for s, n in zip(meta.px_start, meta.n_px)])
    print(f"device: {device}   interior 5x5 patches: {len(nbr_np):,} "
          f"(per leaf min {p_count.min()}, median {int(np.median(p_count))})")
    print("leaves per class: " + ", ".join(f"{c} {(cls == k).sum()}" for k, c in enumerate(CLASSES)))

    P = np.full((len(meta), len(CLASSES)), np.nan)
    stopping = {}
    for fold in folds:
        seed_everything(fold)
        train_idx, test_idx = fold_indices(meta, fold)
        val_fold = (fold + 1) % N_FOLDS
        is_val = meta.fold.to_numpy()[train_idx] == val_fold
        fit_idx, val_idx = train_idx[~is_val], train_idx[is_val]
        lo, hi = float(leaf_lo[train_idx].min()), float(leaf_hi[train_idx].max())
        print(f"fold {fold}: {len(fit_idx)} train / {len(val_idx)} validation (fold {val_fold}) / "
              f"{len(test_idx)} test leaves; 0-1 scaling from [{lo:.4f}, {hi:.4f}]")
        model = PMMFNet(n_out=len(CLASSES), channels=args.channels, n_bands=N_BANDS_IN)
        stopping[fold] = {"val_fold": val_fold, **train_fold(
            model, pix, nbr, p_start, p_count, fit_idx, val_idx, cls, lo, hi,
            args, device, tag=f"{args.name} f{fold}")}
        ids, owner = patch_ids(test_idx, p_start, p_count, device)
        P[test_idx] = leaf_probs(model, pix, nbr, ids, owner, len(test_idx), lo, hi)
        print_fold_summary(args.name, fold, meta, test_idx,
                           P[test_idx, 1] + P[test_idx, 3], P[test_idx, 2] + P[test_idx, 3])
        del model
        torch.cuda.empty_cache()

    config = {"lr": args.lr, "batch": args.batch_size, "K per class": args.samples_per_class,
              "max epochs": args.max_epochs, "patience": args.patience, "channels": args.channels,
              "bands": f"204 zero-filled to {N_BANDS_IN}", "folds": ",".join(map(str, folds))}
    md = save_results(args.name, "11 PMMFNet - out-of-fold results", meta, P, config, args.quick,
                      extra_md=stopping_markdown(stopping), extra_json={"early_stopping": stopping})
    print("\n" + md)

if __name__ == "__main__":
    main()
