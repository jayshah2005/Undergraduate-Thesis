"""APPROACH 12 - GMKL (Hassanzadeh et al., Int. J. Remote Sensing 45:3075-3103, 2024).

Run from claude runs/gmkl/:
       ../../.venv/bin/python -u 12_gmkl.py                      (5 folds; pixel and leaf units, both penalties)
       ../../.venv/bin/python -u 12_gmkl.py --quick              (fold 0, small samples; writes results/quick/)
       ../../.venv/bin/python -u 12_gmkl.py --units leaf --penalties pair

Reads  ../shared/cache (pixels, leaf table, wavelengths, pmmf_pixel_index.npy built by
       ../pmmfnet/11_pmmfnet.py; the superpixel and leaf AS-kernel caches are built here once).
Writes results/12_gmkl__<unit>__<penalty>_penalty_{oof.csv,metrics.md,metrics.json}
       (results/quick/ with --quick) and the run log logs/12.log (logs/quick_12.log with --quick;
       the log is appended to, like the shared logger always does).

The building blocks (superpixels, adjacent-superpixel kernel, attribute profiles, block
diagonal graph, kernel weights) are in gmkl.py. This script is the pipeline around them.
Classes and metrics are shared with 11 (four_class.py): 4 classes healthy / red blotch only
/ leafroll only / both; confusion matrix, balanced accuracy, ROC AUC.

TERMS
  superpixel  a small patch of similar-coloured, connected pixels inside one leaf.
  AS kernel   adjacent-superpixel kernel (Eq. 5): the similarity of two samples is the weighted
              mean Gaussian similarity between the superpixels around each of them, so a pixel
              is described by its neighbourhood, not only by its own (noisy) spectrum.
  EAP         extended attribute profile: a few principal-component images filtered by
              morphological area / standard-deviation filters (spatial texture features).
  base kernel one (n, n) similarity matrix; here 10 AS kernels + 1 EAP kernel = 11.
  eta         the learned non-negative kernel weights (sum 1); the SVM uses sum_m eta_m K_m.
  BDR         block diagonal representation: a graph between sample pairs whose connected
              blocks group pairs that the kernels describe alike (Lu et al. 2019).
  unit        what one SVM sample is: a pixel (pixel unit) or a whole leaf (leaf unit).
  fold        one of the 5 row folds of common.py (fold k = row k+1 of every variety), the
              same test leaves as every other script.

THE PAPER, STEP BY STEP -> WHAT THIS SCRIPT DOES
  RGB bands   3 bands "separated according to their frequency" (ENVI)
              -> the bands nearest 640, 550 and 460 nm, stretched to 0-255 per leaf
                 (1st-99th percentile of the leaf's pixels).
  Superpixels real-time DBSCAN (Shen et al. 2016) at several scales: Indian Pines (145 x 145
              px) used 200, 300, 400, 500, 600 superpixels
              -> same algorithm (gmkl.py), c1 = 0.6, c2 = 0.4, c3 = 1 as in the paper, inside
                 each leaf, with the same pixels per superpixel as those Indian Pines scales:
                 105, 70, 53, 42, 35 px. Colour threshold tc = 30 [unspecified; the smallest
                 of 10/15/20/30/45/70 giving counts within 20% of the target on 30 leaves,
                 no labels used]; regions under 1/4 of the expected size are merged
                 [threshold unspecified].
  AS kernels  Eq. 5 on superpixel mean spectra, Gaussian sigma 0.5 and 1 per scale
              -> same, on all 204 bands of raw reflectance (the paper states no scaling).
                 Neighbour weights w = exp(-2 r0 |.|^2), r0 = 1 / (2 x median squared distance
                 between adjacent superpixels) per scale [r0 unspecified]; a superpixel counts
                 as its own neighbour. 5 scales x 2 sigmas = 10 kernels.
  EAP kernel  3 PCs; area thresholds 1000, 500, 200; standard-deviation thresholds 2.5-10%
              (step 2.5%); stacked; Gaussian kernel with variance 1.5
              -> same; PCA fitted on training pixels of each fold; each PC scaled to [0, 1]
                 by the training 0.5-99.5th percentiles; std thresholds = % of the PC's
                 training mean; outside the leaf the PC image is filled with the leaf mean
                 before filtering. 3 x (1 + 2 x 7) = 45 features. 1 kernel.
  GMKL        Q = [vec(K_1); ...; vec(K_M)] over all N^2 sample pairs; block diagonal graph
              W (Eqs. 14-21); eta = generalised eigenvector with the smallest non-zero
              eigenvalue (Eqs. 10-13)
              -> Q over 2000 sample pairs, half same-class and half different-class, i != j
                 (pixel unit: from different leaves) [all N^2 pairs is billions here];
                 BDR lam = 50, gamma = 1, k = 2 blocks, up to 300 iterations [all unspecified].
                 Penalty graph [undefined in the paper], run both ways:
                   pair : joins same-class pairs to different-class pairs (uses labels)
                   full : complete graph (then the weights use no labels at all)
                 eta: sign so it sums positive, negatives clipped to 0, normalised to sum 1.
  SVM         SVM on the combined kernel; C from 1e2 ... 1e7 by 10-fold cross-validation
              -> same (4 classes, one-vs-one), C chosen by balanced accuracy in a 10-fold
                 inner CV on the training data (fits spread over CPU cores).
                 Decisions (confusion matrix, balanced accuracy) = the SVM's own predicted
                 class; for the pixel unit the majority vote of the leaf's pixels. ROC AUC
                 uses Platt-scaled probabilities (pixel unit: mean over the leaf's pixels).
                 The probabilities are not used for decisions: their argmax ignores the
                 balanced class weights and, in a quick test, never chose LR only or both.
  Unit        pixels of one scene, 1% labelled pixels per class for training
              -> run two ways:
                 pixel : 1000 training pixels per class (each training leaf of a class gives
                         an equal share); inner CV grouped by leaf; every pixel of a test leaf
                         is classified and the leaf gets the mean probability.
                 leaf  : one sample per leaf. AS kernel = Eq. 5 with the whole leaf as the
                         target region and its superpixels as the neighbours (weights from the
                         leaf mean spectrum); EAP = leaf mean of the 45 features; SVM with
                         balanced class weights (the classes are 517 / 226 / 139 / 116 leaves).
  Folds       random 1% split -> the 5 row folds of common.py, like every other script.

LIBRARIES
  PyTorch (GPU) does the heavy kernel algebra (gmkl.py). What stays outside torch, and why, is
  noted next to each call; in short: the SVM is LIBSVM (sklearn SVC with a precomputed kernel,
  the paper's classifier, no torch equivalent); the numpy random generators, sklearn PCA and
  numpy summations are kept because swapping them would change the reported numbers.
"""
import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.svm import SVC

import gmkl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))                     # common.py, four_class.py, logs.py, data cache
from common import CACHE_DIR, fold_indices, load_meta, load_pixels, load_wavelengths, n_folds  # noqa: E402
from four_class import CLASSES, fmt, leaf_classes, save_results                              # noqa: E402
from logs import get_logger                                                                  # noqa: E402

RESULTS = HERE / "results"                                           # this pipeline's outputs (not shared/)
LOGS = HERE / "logs"                                                 # run logs (never inside results/)

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

RGB_NM = (640.0, 550.0, 460.0)
SP_SIZES = (105, 70, 53, 42, 35)              # pixels per superpixel (Indian Pines 200-600 sp)
TC, C1, C2, C3, MERGE_FRAC = 30.0, 0.6, 0.4, 1.0, 0.25
AS_SIGMAS = (0.5, 1.0)
EAP_AREA = (1000, 500, 200)
EAP_STD = (0.025, 0.05, 0.075, 0.10)          # fraction of the PC's mean
EAP_SIGMA2 = 1.5
C_GRID = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
BDR_LAM, BDR_GAMMA, BDR_K, BDR_ITER = 50.0, 1.0, 2, 300
KERNEL_NAMES = [f"AS {s}px s{sg}" for s in SP_SIZES for sg in AS_SIGMAS] + ["EAP"]
DECISION_NOTE = {
    "leaf": "Decisions = the SVM's predicted class; ROC AUC uses its Platt-scaled probabilities.",
    "pixel": "Decisions = the majority vote of the SVM's predicted classes over the leaf's pixels; "
             "ROC AUC uses the mean of the pixels' Platt-scaled probabilities."}


# --------------------------------------------------------------------------
# 1. Superpixels for every leaf at every scale (no labels; built once, cached)
# --------------------------------------------------------------------------
def leaf_rgb(pixels, index, meta, i, rgb_bands):
    """Pseudo-RGB image of leaf i (bands nearest 640 / 550 / 460 nm, 1-99 % stretch) and its mask;
    the input of the DBSCAN superpixels.

    Returns ((H, W, 3) float64 in 0-255, zero outside the leaf; (H, W) bool leaf mask)."""
    h, w = int(meta.height[i]), int(meta.width[i])
    idx = index[i, :h, :w]                                   # (H, W) pixel row in pixels.npy, -1 = background
    m = idx >= 0
    img = np.zeros((h, w, 3))
    img[m] = pixels[idx[m]][:, rgb_bands]
    # Per-leaf robust stretch, so the fixed colour threshold tc means the same on dark and bright
    # leaves. numpy, not torch: feeds the numba superpixel code and the cached superpixels.
    lo, hi = np.percentile(img[m], 1, axis=0), np.percentile(img[m], 99, axis=0)
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255 * m[..., None], m


def build_superpixels(meta, pixels, index, device, log: logging.Logger | None = None) -> list[dict]:
    """Per scale: px_sp (pixel -> superpixel id), sp_leaf, sp_mean (raw reflectance, 204 bands),
    adjacency (CSR indptr / indices) and r0. Cached in cache/gmkl_sp_<size>px_tc<tc>.npz.

    Superpixel ids are global (numbered across all leaves), so one id indexes sp_mean directly.
    log: optional logger for one line per scale that has to be built (slow: minutes per scale)."""
    out = []
    rgb_bands = [int(np.argmin(abs(load_wavelengths() - nm))) for nm in RGB_NM]
    for size in SP_SIZES:
        path = CACHE_DIR / f"gmkl_sp_{size}px_tc{TC:g}.npz"
        if not path.exists():
            t0 = time.time()
            px_sp = np.empty(len(pixels), np.int32)
            sp_leaf, adj, offset = [], [], 0
            for i in range(len(meta)):
                # Superpixels never cross leaves: each leaf is segmented on its own and its labels
                # are shifted by `offset` to make them globally unique.
                rgb, m = leaf_rgb(pixels, index, meta, i, rgb_bands)
                lab = gmkl.dbscan_superpixels(rgb, m, max(1, int(m.sum()) // size), TC, C1, C2, C3, MERGE_FRAC)
                n_sp = int(lab.max()) + 1
                s = int(meta.px_start[i])
                px_sp[s:s + int(meta.n_px[i])] = lab[m] + offset       # row-major, like pixels.npy
                adj += [a + offset for a in gmkl.superpixel_adjacency(lab, n_sp)]
                sp_leaf.append(np.full(n_sp, i, np.int32))
                offset += n_sp
            # Mean spectrum per superpixel: scatter-add every pixel into its superpixel on the GPU
            # (float64, in 1M-pixel chunks to bound GPU memory), then divide by the pixel count.
            sums = torch.zeros(offset, pixels.shape[1], dtype=torch.float64, device=device)
            for s in range(0, len(pixels), 1_000_000):
                sums.index_add_(0, torch.from_numpy(px_sp[s:s + 1_000_000].astype(np.int64)).to(device),
                                torch.from_numpy(pixels[s:s + 1_000_000]).to(device).double())
            sp_size = np.bincount(px_sp, minlength=offset)
            sp_mean = (sums.cpu().numpy() / sp_size[:, None]).astype(np.float32)
            # Adjacency as CSR (indptr / indices) so neighbourhoods() can slice it cheaply.
            indptr = np.concatenate([[0], np.cumsum([len(a) for a in adj])])
            indices = np.concatenate(adj).astype(np.int32)
            # r0 sets how fast a neighbour's weight decays with spectral distance; 1 / (2 x median
            # squared distance between adjacent superpixels) makes a typical neighbour weigh
            # exp(-1). numpy, not torch: one-off, and the result is cached with the superpixels.
            a, b = np.repeat(np.arange(offset), np.diff(indptr)), indices
            d2 = ((sp_mean[a] - sp_mean[b]) ** 2).sum(1)
            np.savez(path, px_sp=px_sp, sp_leaf=np.concatenate(sp_leaf), sp_mean=sp_mean, sp_size=sp_size,
                     indptr=indptr, indices=indices, r0=1.0 / (2.0 * np.median(d2)))
            if log is not None:
                log.info(f"  superpixels {size}px: {offset:,} in {time.time() - t0:.0f}s "
                         f"(per leaf median {int(np.median(np.bincount(np.concatenate(sp_leaf))))})")
        z = np.load(path)
        out.append({k: z[k] for k in z.files} | {"size": size})
    return out


def neighbourhoods(sc: dict, ids: np.ndarray) -> list[np.ndarray]:
    """Each superpixel in ids together with its adjacent superpixels (Eq. 5's set)."""
    return [np.concatenate([[u], sc["indices"][sc["indptr"][u]:sc["indptr"][u + 1]]]) for u in ids]


# --------------------------------------------------------------------------
# 2. Extended attribute profiles for every pixel (per fold: the PCA uses training pixels)
# --------------------------------------------------------------------------
def eap_features(pixels, index, meta, train_idx, seed) -> np.ndarray:
    """(n_pixels, 45) float32: for each of 3 PCs, the PC image plus 7 thinnings and 7 thickenings.

    The PCA and the [0, 1] scaling are fitted on up to 300 random pixels of every *training*
    leaf, so nothing about the test leaves leaks into the features."""
    # numpy RNG + sklearn PCA, not torch: they fix the pixel sample and the PCs of the reported run.
    rng = np.random.default_rng(seed)
    sample = np.concatenate([int(meta.px_start[i]) + rng.choice(int(meta.n_px[i]), min(300, int(meta.n_px[i])),
                                                                  replace=False) for i in train_idx])
    pca = PCA(n_components=3, random_state=seed).fit(pixels[np.sort(sample)])
    pcs_train = pca.transform(pixels[np.sort(sample)])
    # Scaling by robust training percentiles gives every PC the same range, so the paper's fixed
    # area / std thresholds act alike on all three.
    lo, hi = np.percentile(pcs_train, 0.5, axis=0), np.percentile(pcs_train, 99.5, axis=0)
    scaled_mean = np.clip((pcs_train - lo) / (hi - lo), 0, 1).mean(0)
    out = np.empty((len(pixels), 3 * (1 + 2 * (len(EAP_AREA) + len(EAP_STD)))), np.float32)
    for i in range(len(meta)):
        h, w = int(meta.height[i]), int(meta.width[i])
        idx = index[i, :h, :w]
        m = idx >= 0
        pcs = np.clip((pca.transform(pixels[idx[m]]) - lo) / (hi - lo), 0, 1)
        feats = []
        for c in range(3):
            # Background = the leaf's mean so the leaf edge does not look like a huge bright or
            # dark structure to the morphological filters.
            img = np.full((h, w), pcs[:, c].mean())
            img[m] = pcs[:, c]
            ap = gmkl.attribute_profile(img, EAP_AREA, tuple(f * scaled_mean[c] for f in EAP_STD))
            feats.append(ap[:, m].T)
        s = int(meta.px_start[i])
        out[s:s + int(meta.n_px[i])] = np.concatenate(feats, 1)
    return out


# --------------------------------------------------------------------------
# 3. Base kernels for the two units
# --------------------------------------------------------------------------
def leaf_as_kernels(meta, pixels, scales, device) -> np.ndarray:
    """(10, N_leaves, N_leaves) float32: Eq. 5 with each whole leaf as the target region. Cached.

    Uses no labels and no fold information, so one cache serves every fold."""
    path = CACHE_DIR / f"gmkl_leaf_as_tc{TC:g}.npy"
    if path.exists():
        return np.load(path)
    leaf_mean = np.stack([pixels[int(s):int(s) + int(n)].mean(0) for s, n in zip(meta.px_start, meta.n_px)])
    out = []
    for sc in scales:
        # The "neighbourhood" of a leaf is all of its own superpixels, weighted by how close each
        # is to the leaf's mean spectrum (gmkl.neighbour_weights).
        members = [np.where(sc["sp_leaf"] == i)[0] for i in range(len(meta))]
        A = gmkl.neighbour_weights(leaf_mean, members, sc["sp_mean"], float(sc["r0"]))
        G = gmkl.gaussian_sandwich(A, A, sc["sp_mean"], AS_SIGMAS, device)
        out += [G[s].cpu().numpy() for s in AS_SIGMAS]
    out = np.stack(out).astype(np.float32)
    np.save(path, out)
    return out


class PixelASKernels:
    """Eq. 5 kernels between pixels: a pixel's value is that of its superpixel, so everything
    is computed between superpixels and looked up per pixel.

    Why: Eq. 5 depends on a pixel only through its superpixel (centre = superpixel mean,
    neighbours = adjacent superpixels), so all pixels of a superpixel share one kernel row.
    Computing superpixel x superpixel and indexing is far cheaper than pixel x pixel."""

    def __init__(self, scales, train_px, device):
        self.scales, self.device, self.train_px = scales, device, train_px
        self.A_train, self.pos_train = [], []
        for sc in scales:
            # u = distinct superpixels holding training pixels; pos maps each training pixel to its row in u.
            u, pos = np.unique(sc["px_sp"][train_px], return_inverse=True)
            self.A_train.append(gmkl.neighbour_weights(sc["sp_mean"][u], neighbourhoods(sc, u),
                                                       sc["sp_mean"], float(sc["r0"])))
            self.pos_train.append(pos)

    def train(self) -> list[np.ndarray]:
        """10 (n_train, n_train) float32 kernels between the training pixels (5 scales x 2 sigmas)."""
        out = []
        for sc, A, pos in zip(self.scales, self.A_train, self.pos_train):
            G = gmkl.gaussian_sandwich(A, A, sc["sp_mean"], AS_SIGMAS, self.device)
            out += [G[s][pos][:, pos].cpu().numpy() for s in AS_SIGMAS]
        return out

    def prepare_test(self, test_leaves):
        """Kernels between every superpixel of the test leaves and the training pixels (GPU)."""
        self.test_rows = []
        for sc, A, pos in zip(self.scales, self.A_train, self.pos_train):
            v = np.where(np.isin(sc["sp_leaf"], test_leaves))[0]
            Av = gmkl.neighbour_weights(sc["sp_mean"][v], neighbourhoods(sc, v), sc["sp_mean"], float(sc["r0"]))
            G = gmkl.gaussian_sandwich(Av, A, sc["sp_mean"], AS_SIGMAS, self.device)
            lookup = np.full(len(sc["sp_leaf"]), -1, np.int64)   # global superpixel id -> row of G
            lookup[v] = np.arange(len(v))
            pos_t = torch.from_numpy(pos).to(self.device)
            self.test_rows.append((lookup, {s: G[s][:, pos_t] for s in AS_SIGMAS}))

    def test(self, px: np.ndarray) -> list[torch.Tensor]:
        """10 (len(px), n_train) GPU tensors: kernel rows of the test pixels px (call prepare_test first)."""
        out = []
        for sc, (lookup, G) in zip(self.scales, self.test_rows):
            rows = torch.from_numpy(lookup[sc["px_sp"][px]]).to(self.device)
            out += [G[s][rows] for s in AS_SIGMAS]
        return out


def kernel_spread(Ks) -> dict:
    """5th / 50th / 95th percentile of the off-diagonal values of each base kernel."""
    out = {}
    for name, K in zip(KERNEL_NAMES, Ks):
        off = K[~np.eye(len(K), dtype=bool)]
        # numpy, not torch: torch.quantile caps the input size (the pixel unit has 16M values).
        out[name] = np.percentile(off, [5, 50, 95]).round(4).tolist()
    return out


# --------------------------------------------------------------------------
# 4. GMKL weights and the SVM
# --------------------------------------------------------------------------
def sample_pairs(y, groups, n_pairs, rng):
    """n_pairs/2 same-class and n_pairs/2 different-class pairs (i != j, different groups).

    Returns (i, j, same): (n_pairs,) sample indices and a bool flag, same-class pairs first.
    Pairs from the same group (leaf) are skipped: two pixels of one leaf are trivially similar
    and would teach the weights nothing about the classes."""
    # numpy RNG, not torch: keeps the reported pair sample (same random stream).
    same, diff = [], []
    while len(same) < n_pairs // 2 or len(diff) < n_pairs // 2:
        i, j = rng.integers(0, len(y), 2 * n_pairs), rng.integers(0, len(y), 2 * n_pairs)
        ok = (i != j) & (groups[i] != groups[j])
        for a, b in zip(i[ok], j[ok]):
            (same if y[a] == y[b] else diff).append((a, b))
    p = np.array(same[:n_pairs // 2] + diff[:n_pairs // 2])
    return p[:, 0], p[:, 1], np.r_[np.ones(n_pairs // 2, bool), np.zeros(n_pairs // 2, bool)]


def _cv_score(K, y, tr, va, C, class_weight):
    """Balanced accuracy of an SVM with precomputed kernel K, trained on tr and scored on va.

    Top-level (not a closure) so joblib can send it to worker processes."""
    # sklearn SVC (LIBSVM) with a precomputed kernel: the paper's classifier; torch has no equivalent.
    svc = SVC(kernel="precomputed", C=C, class_weight=class_weight).fit(K[np.ix_(tr, tr)], y[tr])
    return balanced_accuracy_score(y[va], svc.predict(K[np.ix_(va, tr)]))


def choose_c(K, y, groups, class_weight, n_splits, seed, n_jobs) -> tuple[float, dict]:
    """C with the best mean balanced accuracy in an inner CV (ties -> smaller C).
    Pixel unit (class_weight None): folds grouped by leaf; leaf unit: stratified folds.

    Returns (best C, {C: mean inner-CV balanced accuracy})."""
    if class_weight is None:
        # Grouped by leaf: pixels of one leaf are near-duplicates, so splitting a leaf between
        # inner-train and inner-validation would reward overfitting (a large C).
        splits = list(GroupKFold(n_splits=n_splits).split(K, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(K, y))
    # Every (C, split) fit is independent -> spread over CPU processes.
    tasks = [(C, tr, va) for C in C_GRID for tr, va in splits]
    accs = Parallel(n_jobs=n_jobs)(delayed(_cv_score)(K, y, tr, va, C, class_weight) for C, tr, va in tasks)
    scores = {C: float(np.mean([a for (c, _, _), a in zip(tasks, accs) if c == C])) for C in C_GRID}
    # Rounding to 6 decimals makes equal scores tie exactly; a tie then goes to the smaller C
    # (stronger regularisation, the safer choice).
    best = max(C_GRID, key=lambda c: (round(scores[c], 6), -c))
    return best, scores


def fit_gmkl(Ks, y, groups, n_pairs, seed, device):
    """BDR graph on sampled pairs; eta per penalty. Returns ({penalty: eta}, info dict)."""
    rng = np.random.default_rng(seed)
    i, j, same = sample_pairs(y, groups, n_pairs, rng)
    # Q (M kernels, n_pairs): column = the M kernel values of one sample pair. The paper uses
    # every pair (vec(K_m)); a few thousand sampled pairs keep the n_pairs x n_pairs graph small.
    Q = np.stack([K[i, j] for K in Ks]).astype(np.float64)
    t0 = time.time()
    W, bdr_info = gmkl.block_diagonal_graph(Q, BDR_LAM, BDR_GAMMA, BDR_K, BDR_ITER, device=device)
    etas, info = {}, {"bdr": bdr_info | {"seconds": round(time.time() - t0, 1)}}
    for penalty in ("pair", "full"):
        etas[penalty], info[penalty] = gmkl.gda_kernel_weights(Q, W, same, penalty)
    return etas, info


# --------------------------------------------------------------------------
# 5. One fold of each unit
# --------------------------------------------------------------------------
def run_leaf_fold(fold, meta, cls, leaf_as, eap, penalties, args, device):
    """One test fold at leaf level: learn the kernel weights on training leaves, pick C, predict test leaves.

    Returns (test_idx, {penalty: (P (n_test, 4) probabilities, (n_test,) predicted class, record)},
    info) where info holds the BDR summary and the kernel spread."""
    train_idx, test_idx = fold_indices(meta, fold)
    # numpy mean, not torch: numpy's pairwise float32 sum keeps the reported leaf features.
    leaf_eap = np.stack([eap[int(s):int(s) + int(n)].mean(0) for s, n in zip(meta.px_start, meta.n_px)])
    E = torch.from_numpy(leaf_eap).float().to(device)
    K_all = list(leaf_as) + [gmkl.gaussian(E, E, EAP_SIGMA2).cpu().numpy()]
    Ks_tr = [K[np.ix_(train_idx, train_idx)] for K in K_all]
    y = cls[train_idx]
    # groups = leaf ids, so every sampled pair joins two different leaves
    etas, info = fit_gmkl(Ks_tr, y, train_idx, args.pairs, fold, device)
    info["kernel_spread"] = kernel_spread(Ks_tr)
    res = {}
    for pen in penalties:
        # Combined kernel over all leaves (train and test). numpy, not torch: float64 eta x float32
        # K promotes exactly as reported, and SVC needs numpy anyway.
        Kc = sum(e * K for e, K in zip(etas[pen], K_all))
        C, cv = choose_c(Kc[np.ix_(train_idx, train_idx)], y, train_idx, "balanced", args.inner_folds, fold,
                         args.jobs)
        # probability=True adds Platt scaling (used for ROC AUC only); seeded for reproducibility.
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced", probability=True,
                  random_state=fold).fit(Kc[np.ix_(train_idx, train_idx)], y)
        P = svc.predict_proba(Kc[np.ix_(test_idx, train_idx)])
        res[pen] = (P, svc.predict(Kc[np.ix_(test_idx, train_idx)]),
                    {"C": C, "cv_balanced_accuracy": cv, "eta": etas[pen].round(4).tolist(), **info[pen]})
    return test_idx, res, info


def sample_train_pixels(meta, train_idx, cls, per_class, rng):
    """per_class pixels per class, each training leaf of the class giving an equal share.

    Equal shares per leaf stop large leaves from dominating; the remainder per_class % n_leaves
    goes to randomly chosen leaves. Returns (pixel rows, owning leaf of each pixel)."""
    # numpy RNG, not torch: keeps the reported training-pixel sample (same random stream).
    px, owner = [], []
    for c in range(len(CLASSES)):
        leaves = train_idx[cls[train_idx] == c]
        share = np.full(len(leaves), per_class // len(leaves))
        share[rng.choice(len(leaves), per_class % len(leaves), replace=False)] += 1
        for leaf, k in zip(leaves, share):
            if k:
                n = int(meta.n_px[leaf])
                px.append(int(meta.px_start[leaf]) + rng.choice(n, min(k, n), replace=False))
                owner.append(np.full(min(k, n), leaf))
    return np.concatenate(px), np.concatenate(owner)


def run_pixel_fold(fold, meta, cls, scales, eap, penalties, args, device):
    """One test fold at pixel level: sampled training pixels -> kernel weights and C -> every test
    pixel predicted, leaf = majority vote / mean probability.

    Returns the same (test_idx, res, info) as run_leaf_fold; the probabilities are leaf means."""
    train_idx, test_idx = fold_indices(meta, fold)
    rng = np.random.default_rng(fold)
    train_px, owner = sample_train_pixels(meta, train_idx, cls, args.pixels_per_class, rng)
    y = cls[owner]
    # 1. base kernels between the training pixels -> kernel weights eta (pairs from different leaves)
    as_k = PixelASKernels(scales, train_px, device)
    E_tr = torch.from_numpy(eap[train_px]).float().to(device)
    Ks_tr = as_k.train() + [gmkl.gaussian(E_tr, E_tr, EAP_SIGMA2).cpu().numpy()]
    etas, info = fit_gmkl(Ks_tr, y, owner, args.pairs, fold, device)
    info["kernel_spread"] = kernel_spread(Ks_tr)
    info["n_train_pixels"] = int(len(train_px))

    # 2. test side: every pixel of every test leaf, in chunks of args.chunk pixels
    as_k.prepare_test(test_idx)
    test_px = np.concatenate([np.arange(int(meta.px_start[i]), int(meta.px_start[i]) + int(meta.n_px[i]))
                              for i in test_idx])
    test_owner = np.repeat(np.arange(len(test_idx)), meta.n_px.to_numpy()[test_idx])   # position in test_idx
    res = {}
    for pen in penalties:
        eta = etas[pen]
        # numpy, not torch: float64 eta x float32 K promotes exactly as reported, and SVC needs numpy.
        Kc = sum(e * K for e, K in zip(eta, Ks_tr))
        C, cv = choose_c(Kc, y, owner, None, args.inner_folds, fold, args.jobs)
        svc = SVC(kernel="precomputed", C=C, probability=True, random_state=fold).fit(Kc, y)
        # Pixel -> leaf: sum each leaf's pixel probabilities (mean, for ROC AUC) and count its
        # pixels' SVM votes (majority, for the decision). The probabilities' argmax is not used
        # for decisions because Platt scaling ignores the class balance (see the file docstring).
        sums, votes, t0 = np.zeros((len(test_idx), len(CLASSES))), np.zeros((len(test_idx), len(CLASSES))), time.time()
        for s in range(0, len(test_px), args.chunk):
            px = test_px[s:s + args.chunk]
            E = torch.from_numpy(eap[px]).float().to(device)
            parts = as_k.test(px) + [gmkl.gaussian(E, E_tr, EAP_SIGMA2)]
            Kt = sum(float(e) * K for e, K in zip(eta, parts)).double().cpu().numpy()   # combined kernel on GPU
            # np.add.at, not torch scatter: in-order accumulation keeps the reported sums bit-for-bit.
            np.add.at(sums, test_owner[s:s + args.chunk], svc.predict_proba(Kt))
            np.add.at(votes, (test_owner[s:s + args.chunk], svc.predict(Kt)), 1)
        n = meta.n_px.to_numpy()[test_idx][:, None]
        res[pen] = (sums / n, votes.argmax(1),
                    {"C": C, "cv_balanced_accuracy": cv, "eta": eta.round(4).tolist(),
                     "test_seconds": round(time.time() - t0, 1), **info[pen]})
    return test_idx, res, info


def run_folds(meta, cls, pixels, index, scales, leaf_as, folds, units, penalties, args, device,
              log: logging.Logger):
    """Run every fold for each unit; the test leaves of each fold fill the out-of-fold arrays.

    Returns (P {(unit, penalty): (N_leaves, 4) probabilities, NaN = not predicted},
             pred {(unit, penalty): (N_leaves,) predicted class, -1 = not predicted},
             records {(unit, penalty): {fold: per-fold record for the report}}).
    log receives the per-fold progress lines."""
    P = {(u, p): np.full((len(meta), len(CLASSES)), np.nan) for u in units for p in penalties}
    pred = {(u, p): np.full(len(meta), -1) for u in units for p in penalties}
    records = {(u, p): {} for u in units for p in penalties}
    for fold in folds:
        tf = time.time()
        train_idx, _ = fold_indices(meta, fold)
        # The EAP PCA is refitted per fold on training pixels only; both units share it.
        eap = eap_features(pixels, index, meta, train_idx, fold)
        log.info(f"fold {fold}: attribute profiles ({time.time() - tf:.0f}s)")
        for unit in units:
            tu = time.time()
            run = run_leaf_fold if unit == "leaf" else run_pixel_fold
            test_idx, res, info = run(fold, meta, cls, leaf_as if unit == "leaf" else scales, eap,
                                      penalties, args, device)
            spread = "; ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for k, v in info["kernel_spread"].items())
            log.info(f"  [{unit}] kernel spread p5/p50/p95: {spread}")
            log.info(f"  [{unit}] BDR: {info['bdr']}")
            for pen, (Pf, pf, rec) in res.items():
                P[(unit, pen)][test_idx] = Pf
                pred[(unit, pen)][test_idx] = pf
                records[(unit, pen)][fold] = rec | {"bdr": info["bdr"], "kernel_spread": info["kernel_spread"]}
                counts = np.bincount(Pf.argmax(1), minlength=len(CLASSES))
                votes = np.bincount(pf, minlength=len(CLASSES))
                log.info(f"  [{unit}/{pen}] C={rec['C']:g}  eta={np.round(rec['eta'], 2).tolist()}  "
                         f"predicted classes (prob argmax) {counts.tolist()}, (SVM decision) {votes.tolist()}  "
                         f"test bal.acc (SVM decision) {balanced_accuracy_score(cls[test_idx], pf):.3f}")
            log.info(f"  [{unit}] done in {time.time() - tu:.0f}s")
    return P, pred, records


# --------------------------------------------------------------------------
# 6. Reports
# --------------------------------------------------------------------------
def fold_markdown(records: dict) -> str:
    """Per-fold table of the report: chosen C, inner-CV score, BDR iterations, learned kernel weights."""
    L = ["", "## GMKL per fold", "",
         "eta = learned kernel weights, in the order " + ", ".join(KERNEL_NAMES) + ".", "",
         "| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |",
         "|---|---|---|---|---|---|"]
    for fold, r in records.items():
        L.append(f"| {fold} | {r['C']:g} | {fmt(r['cv_balanced_accuracy'][r['C']])} | "
                 f"{r['bdr']['iterations']} ({r['bdr']['converged']}) | {r['n_clipped']}"
                 f"{' (uniform fallback)' if r['uniform_fallback'] else ''} | "
                 + " ".join(f"{e:.2f}" for e in r["eta"]) + " |")
    return "\n".join(L) + "\n"


def write_reports(meta, P, pred, records, folds, units, penalties, args, log: logging.Logger) -> None:
    """Write <name>_oof.csv / _metrics.md / _metrics.json per unit and penalty (four_class.save_results)
    into results/ (results/quick/ with --quick) and log each markdown report."""
    for unit in units:
        for pen in penalties:
            name = f"12_gmkl__{unit}__{pen}_penalty"
            config = {"unit": unit, "penalty": pen, "superpixel px": "/".join(map(str, SP_SIZES)),
                      "tc": TC, "pairs": args.pairs, "inner CV folds": args.inner_folds,
                      "folds": ",".join(map(str, folds))}
            if unit == "pixel":
                config["train pixels per class"] = args.pixels_per_class
            md = save_results(name, f"12 GMKL ({unit} unit, {pen} penalty) - out-of-fold results",
                              meta, P[(unit, pen)], config, args.quick,
                              extra_md=fold_markdown(records[(unit, pen)]),
                              extra_json={"gmkl": records[(unit, pen)], "kernel_names": KERNEL_NAMES},
                              pred=pred[(unit, pen)], decision_note=DECISION_NOTE[unit],
                              folder=RESULTS / "quick" if args.quick else RESULTS)
            log.info("\n" + md)


def parse_args() -> argparse.Namespace:
    """Command line; --quick shrinks the samples (fold 0, 100 px per class, 400 pairs, 3 inner folds)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--units", default="pixel,leaf")
    ap.add_argument("--penalties", default="pair,full")
    ap.add_argument("--pixels-per-class", type=int, default=1000)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--inner-folds", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="CPU processes for the inner-CV SVM fits")
    ap.add_argument("--folds", default=None, help="comma-separated, e.g. 0,1 (default: all)")
    args = ap.parse_args()
    if args.quick:
        args.pixels_per_class, args.pairs, args.inner_folds = 100, 400, 3
    return args


def main():
    """Parse the command line, set up the log, build the caches, run every fold, write the reports."""
    args = parse_args()
    folds = [0] if args.quick else ([int(k) for k in args.folds.split(",")] if args.folds
                                    else range(n_folds()))
    units, penalties = args.units.split(","), args.penalties.split(",")
    log = get_logger("gmkl", log_file=LOGS / ("quick_12.log" if args.quick else "12.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False           # distances need full fp32 (|x|^2 + |y|^2 - 2xy cancels)

    t0 = time.time()
    meta = load_meta()
    cls = leaf_classes(meta)
    pixels = load_pixels(mmap=False)
    index = np.load(CACHE_DIR / "pmmf_pixel_index.npy")    # built by ../pmmfnet/11_pmmfnet.py
    scales = build_superpixels(meta, pixels, index, device, log)
    leaf_as = leaf_as_kernels(meta, pixels, scales, device) if "leaf" in units else None
    log.info(f"data, superpixels{' and leaf AS kernels' if leaf_as is not None else ''} ready "
             f"({time.time() - t0:.0f}s); r0 per scale: " + ", ".join(f"{float(s['r0']):.1f}" for s in scales))

    P, pred, records = run_folds(meta, cls, pixels, index, scales, leaf_as, folds, units, penalties, args,
                                 device, log)
    write_reports(meta, P, pred, records, folds, units, penalties, args, log)
    log.info(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
