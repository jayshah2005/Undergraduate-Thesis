"""Red blotch from leaf spectra within Chardonnay: feature sets × classical models × fold schemes."""
from __future__ import annotations

import os

# One BLAS thread per process: the permutation test runs many small fits in parallel workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from common import H5_PATH, load_meta, load_wavelengths  # noqa: E402
from logs import get_logger  # noqa: E402
from spectra import leaf_mean_spectra, snv  # noqa: E402

VARIETY = "Chardonnay"
ROWS = ["R1", "R2", "R3", "R4", "R5"]
SCHEMES = {
    "row": "fold k = field row k (unseen rows)",
    "photo": "random photos dealt evenly per row, 2 per fold; R1 has 13 photos, so 3 or 2 (new vines in seen rows)",
    "block": "fold k = the photos of each row whose first vine is in 10k+1 to 10k+10 "
             "(held-out vines are neighbours of each other)",
}
FOLD_SEED = 0              # same photo assignment as clean/disease_classification/attention_mil (verified identical)
PERM_SCHEME = "block"      # the pre-declared scheme for the primary comparison
PRIMARY = ("D", "logreg")
BASELINE = ("A", "logreg_c1")
N_BOOT, BOOT_SEED, PERM_SEED = 2000, 0, 1

EDGE_ERODE = 2             # px removed from the leaf margin before pixel statistics
GLINT = 1.0                # a pixel above this in any band is a specular glint
R_MIN = 1e-3               # reflectance floor before indices that divide by it

C_GRID = (0.01, 0.1, 1.0, 10.0)
PLS_GRID = (1, 2, 3, 5, 8, 12)
HGB_GRID = ((2, 50), (2, 150), (3, 50), (3, 150))   # (depth, trees)

# Pipelines: (feature set, model). The baseline is the reported one, C fixed at 1.
PIPELINES = [("A", "logreg_c1"), ("A", "logreg"), ("A", "pls"), ("B", "logreg"), ("B", "pls"),
             ("C", "logreg"), ("C", "pls"), ("C", "hgb"), ("D", "logreg"), ("D", "pls"), ("D", "hgb"),
             ("E", "logreg"), ("E", "pls"), ("E", "hgb"), ("E", "single")]
LINEAR = [p for p in PIPELINES if p[1] != "hgb"]
SET_LABEL = {"A": "A mean SNV spectrum", "B": "B mean 1st derivative", "C": "C indices (leaf mean)",
             "D": "D index spread (pixels)", "E": "E = C + D"}
MODEL_LABEL = {"logreg_c1": "logistic, C=1 (baseline)", "logreg": "logistic", "pls": "PLS-DA",
               "hgb": "gradient boosting", "single": "best single feature"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SCHEME_COLORS = {"row": "#2a78d6", "photo": "#eb6834", "block": "#1baf7a"}


def pname(p: tuple) -> str:
    """Readable pipeline name, e.g. "D index spread (pixels) · logistic"."""
    return f"{SET_LABEL[p[0]]} · {MODEL_LABEL[p[1]]}"


# --------------------------------------------------------------------------
# 1. Features
# --------------------------------------------------------------------------
def vegetation_indices(R: torch.Tensor, w: np.ndarray) -> dict[str, torch.Tensor]:
    """Eleven vegetation indices from reflectance spectra; NaN where a denominator is ~0.

    Each index uses the band nearest the named wavelength (bands are ~3 nm apart).
    NDVI, NDRE, CI red-edge, red-edge position (REP): chlorophyll and the red edge.
    PRI: xanthophyll / photosynthetic stress. ARI: anthocyanins. CRI: carotenoids.
    MCARI: chlorophyll absorption. SIPI: carotenoid/chlorophyll ratio.
    WI (900/970 nm): water. GR (550/670 nm): green vs red.

    n: number of spectra.

    :param R: Reflectance, shape (n, 204).
    :type R: torch.Tensor
    :param w: Band centres in nm, shape (204,).
    :type w: np.ndarray
    :return: {index name: values, shape (n,)}.
    :rtype: dict[str, torch.Tensor]
    """
    R = R.clamp_min(R_MIN)
    b = {nm: R[:, int(np.abs(w - nm).argmin())] for nm in
         (445, 510, 531, 550, 570, 670, 680, 700, 720, 740, 780, 790, 800, 900, 970)}

    def ratio(num, den):
        return torch.where(den.abs() > 1e-4, num / den, torch.full_like(num, float("nan")))

    return {
        "NDVI": ratio(b[800] - b[670], b[800] + b[670]),
        "NDRE": ratio(b[790] - b[720], b[790] + b[720]),
        "CIre": ratio(b[790], b[720]) - 1,
        "REP": 700 + 40 * ratio((b[670] + b[780]) / 2 - b[700], b[740] - b[700]),
        "PRI": ratio(b[531] - b[570], b[531] + b[570]),
        "ARI": 1 / b[550] - 1 / b[700],
        "CRI": 1 / b[510] - 1 / b[550],
        "MCARI": ((b[700] - b[670]) - 0.2 * (b[700] - b[550])) * ratio(b[700], b[670]),
        "SIPI": ratio(b[800] - b[445], b[800] - b[680]),
        "WI": ratio(b[900], b[970]),
        "GR": ratio(b[550], b[670]),
    }


def interior_pixels(cube: np.ndarray, device: torch.device) -> torch.Tensor:
    """Leaf pixels away from the margin and without glints.

    The leaf mask is eroded by EDGE_ERODE px: margin pixels mix leaf and background,
    and a small leaf has relatively more of them, so they would make the low
    percentiles a leaf-size measure.

    lines, samples: spatial size of the cube. m: interior pixels kept.

    :param cube: Reflectance as stored in the HDF5, shape (lines, 204, samples).
    :type cube: np.ndarray
    :param device: GPU or CPU.
    :type device: torch.device
    :return: Pixels, shape (m, 204) float32.
    :rtype: torch.Tensor
    """
    x = torch.from_numpy(np.nan_to_num(cube, nan=0.0)).to(device).permute(1, 0, 2)   # (204, H, W)
    leaf = (x != 0).any(0).float()[None, None]
    k = 2 * EDGE_ERODE + 1
    inner = -F.max_pool2d(-F.pad(leaf, (EDGE_ERODE,) * 4, value=0.0), k, 1)        # min-pool = erosion
    keep = (inner[0, 0] > 0) & ~(x > GLINT).any(0)
    return x[:, keep].T


def build_features(meta: pd.DataFrame, w: np.ndarray, device: torch.device) -> dict:
    """Compute feature sets A-E and the leaf-size measures for every leaf of meta.

    n: leaves (250).

    :param meta: Chardonnay leaf table (path, row, vine, labels, n_px), n rows.
    :type meta: pandas.DataFrame
    :param w: Band centres in nm, shape (204,).
    :type w: np.ndarray
    :param device: GPU or CPU.
    :type device: torch.device
    :return: {"A".."E": (n, d) float64, "names_C", "names_D": feature names, "n_px", "n_interior": (n,)}.
    :rtype: dict
    """
    from scipy.signal import savgol_filter  # why not torch: standard filter on a (250, 204) array

    full_meta = load_meta()
    means, _glint = leaf_mean_spectra(full_meta)
    X = means[(full_meta.variety == VARIETY).to_numpy()]
    A = snv(X)
    B = savgol_filter(A, 11, 2, deriv=1, axis=1)

    ci = vegetation_indices(torch.tensor(X, device=device), w)
    names_C = list(ci)
    C = torch.stack([ci[k] for k in names_C], 1).double().cpu().numpy()

    rows_D, n_int = [], []
    with h5py.File(H5_PATH, "r") as h5:
        for path in meta.path:
            px = interior_pixels(np.asarray(h5[path]["cube"], dtype=np.float32), device)
            n_int.append(len(px))
            vi = vegetation_indices(px, w)
            feats = []
            for k in names_C:
                q = torch.nanquantile(vi[k].double(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9],
                                                                   dtype=torch.float64, device=device))
                feats += [q[0], q[2], q[4], q[3] - q[1]]       # p10, p50, p90, IQR
            rows_D.append(torch.stack(feats).cpu().numpy())
    names_D = [f"{k} {s}" for k in names_C for s in ("p10", "p50", "p90", "IQR")]
    D = np.stack(rows_D)
    return {"A": A, "B": B, "C": C, "D": D, "E": np.hstack([C, D]), "names_C": names_C,
            "names_D": names_D, "n_px": meta.n_px.to_numpy(float), "n_interior": np.array(n_int, float)}


def load_features(meta: pd.DataFrame, w: np.ndarray, device: torch.device) -> dict:
    """Read the feature table from cache/ if it matches these leaves and wavelengths, else build and save it.

    The wavelengths are part of the check because the indices pick bands by wavelength:
    a corrected wavelength grid must rebuild the cache, not reuse it.
    """
    path = HERE / "cache" / "leaf_features.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if ("wavelengths" in d.files and list(d["path"]) == list(meta.path)
                and np.array_equal(d["wavelengths"], w)):
            return {k: (list(d[k]) if k.startswith("names") else d[k]) for k in d.files
                    if k not in ("path", "wavelengths")}
    f = build_features(meta, w, device)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, path=np.array(meta.path), wavelengths=w, **f)
    return f


# --------------------------------------------------------------------------
# 2. Folds and the within-row AUC
# --------------------------------------------------------------------------
def fold_of(meta: pd.DataFrame, scheme: str) -> np.ndarray:
    """Fold 0-4 per leaf. Photo = the camera image a leaf was cut from; a photo never splits.

    clean/leaf_sources.csv has each leaf's capture, matched by exact pixel values.
    Rows R2-R5 have 10 photos of 5 leaves; R1 has 12 of 4 and one of 2.
    Same assignment as clean/disease_classification/attention_mil/train.py (checked leaf by leaf).

    :param meta: Chardonnay leaf table, n rows.
    :type meta: pandas.DataFrame
    :param scheme: "row", "photo" or "block".
    :type scheme: str
    :return: Fold per leaf, length n.
    :rtype: np.ndarray
    """
    rows = meta.row.to_numpy()
    if scheme == "row":
        return np.array([ROWS.index(r) for r in rows])
    table = pd.read_csv(H5_PATH.parent / "clean" / "leaf_sources.csv")
    parts = table.leaf.str.split("/", expand=True)
    capture = dict(zip(parts[0] + "/" + parts[1] + "/V" + parts[2], table.capture_id))
    photo = np.array([capture[p] for p in meta.path])
    if scheme == "block":
        first_vine = meta.vine.groupby(photo).transform("min").to_numpy()
        return (first_vine - 1) // 10
    g = torch.Generator().manual_seed(FOLD_SEED)
    fold = np.zeros(len(meta), dtype=int)
    for r in ROWS:
        row_photos = np.unique(photo[rows == r])
        for pos, i in enumerate(torch.randperm(len(row_photos), generator=g).numpy()):
            fold[photo == row_photos[i]] = pos % 5
    return fold


def within_group_auc(s: np.ndarray, y: np.ndarray, group: np.ndarray,
                     w: torch.Tensor | None = None) -> float | torch.Tensor:
    """ROC AUC over (positive, negative) pairs from the same group only.

    group = fold × row: only leaves scored by the same model and from the same row
    are compared, so neither fold-to-fold score shifts nor a row's disease rate count.
    Groups without both classes add no pairs. With bootstrap counts w (B, n), returns
    one AUC per resample.

    :param s: Scores, length n.
    :type s: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param group: Group id per leaf, length n.
    :type group: np.ndarray
    :param w: Optional resampling counts, shape (B, n).
    :type w: torch.Tensor | None
    :return: AUC (float), or shape (B,) with w.
    :rtype: float | torch.Tensor
    """
    s_t = torch.as_tensor(s, dtype=torch.float64)
    num = den = 0.0
    for k in np.unique(group):
        m = group == k
        pos, neg = s_t[m & (y == 1)], s_t[m & (y == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        tab = (pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()
        if w is None:
            num, den = num + tab.sum(), den + tab.numel()
        else:
            wp, wn = w[:, m & (y == 1)], w[:, m & (y == 0)]
            num = num + ((wp @ tab) * wn).sum(1)
            den = den + wp.sum(1) * wn.sum(1)
    return float(num / den) if w is None else num / den


def bootstrap_weights(y: np.ndarray, group: np.ndarray, n_boot: int, seed: int) -> torch.Tensor:
    """Resample leaves with replacement inside each (group, class) cell; same draws for every pipeline.

    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param group: Group per leaf, length n.
    :type group: np.ndarray
    :param n_boot: Number of resamples B.
    :type n_boot: int
    :param seed: Generator seed.
    :type seed: int
    :return: Counts, shape (B, n) float64.
    :rtype: torch.Tensor
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(n_boot, len(y), dtype=torch.float64)
    for k in np.unique(group):
        for c in (0, 1):
            idx = torch.as_tensor(np.flatnonzero((group == k) & (y == c)))
            if len(idx):
                draw = idx[torch.randint(len(idx), (n_boot, len(idx)), generator=g)]
                w.scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
    return w


# --------------------------------------------------------------------------
# 3. Models with inner-CV tuning
# --------------------------------------------------------------------------
def make_model(kind: str, param, n_features: int):
    """Build one classifier.

    Why scikit-learn: exact, well-tested solvers for these classical models, and the
    baseline must match the logistic regression reported before.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind in ("logreg", "logreg_c1"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=param, class_weight="balanced", max_iter=5000))
    if kind == "pls":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=min(param, n_features)))
    depth, trees = param
    return HistGradientBoostingClassifier(max_depth=depth, max_iter=trees, learning_rate=0.05,
                                          class_weight="balanced", random_state=0)


def grid_of(kind: str, n_features: int) -> list:
    """Settings the inner CV chooses from."""
    if kind == "logreg_c1":
        return [1.0]
    if kind == "logreg":
        return list(C_GRID)
    if kind == "pls":
        return [k for k in PLS_GRID if k <= n_features]
    return list(HGB_GRID)


def fit_score(kind: str, param, X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Fit on tr and return a score for te (higher = more likely red blotch)."""
    m = make_model(kind, param, X.shape[1]).fit(X[tr], y[tr])
    return m.predict(X[te]).ravel() if kind == "pls" else m.predict_proba(X[te])[:, 1]


def run_pipeline(X: np.ndarray, y: np.ndarray, fold: np.ndarray, group: np.ndarray,
                 kind: str) -> tuple[np.ndarray, list]:
    """Out-of-fold scores for one pipeline; settings chosen by an inner CV in each training split.

    Inner CV: each of the 4 training folds is held out once; the setting with the
    best within-row AUC over those held-out predictions is refitted on all 4.

    n: leaves.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param fold: Outer fold per leaf, length n.
    :type fold: np.ndarray
    :param group: fold × row id per leaf, length n.
    :type group: np.ndarray
    :param kind: "logreg_c1", "logreg", "pls", "hgb" or "single".
    :type kind: str
    :return: scores (n,) and the chosen setting per outer fold.
    :rtype: tuple[np.ndarray, list]
    """
    if kind == "single":
        return _single_feature(X, y, fold, group)
    grid = grid_of(kind, X.shape[1])
    scores, chosen = np.zeros(len(y)), []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        best = grid[0]
        if len(grid) > 1:
            inner_auc = []
            for param in grid:
                s = np.zeros(len(y))
                for j in sorted(set(fold[tr])):
                    s[fold == j] = fit_score(kind, param, X, y, tr[fold[tr] != j], np.flatnonzero(fold == j))
                inner_auc.append(within_group_auc(s[tr], y[tr], group[tr]))
            best = grid[int(np.argmax(inner_auc))]
        scores[te] = fit_score(kind, best, X, y, tr, te)
        chosen.append(best)
    return scores, chosen


def _single_feature(X: np.ndarray, y: np.ndarray, fold: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, list]:
    """Pick the one feature (and its sign) with the best within-row AUC on the training folds.

    The fairest test of "one index is enough": the feature is chosen without the
    test fold, then used as the score as it is (no fitting beyond the sign).

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param fold: Outer fold per leaf, length n.
    :type fold: np.ndarray
    :param group: fold × row id per leaf, length n.
    :type group: np.ndarray
    :return: scores (n,) and the chosen column index per outer fold (negative = sign flipped).
    :rtype: tuple[np.ndarray, list]
    """
    scores, chosen = np.zeros(len(y)), []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        auc = np.array([within_group_auc(X[tr, j], y[tr], group[tr]) for j in range(X.shape[1])])
        j = int(np.argmax(np.abs(auc - 0.5)))
        sign = 1.0 if auc[j] >= 0.5 else -1.0
        scores[te] = sign * X[te, j]
        chosen.append(f"{'+' if sign > 0 else '−'}{j}")
    return scores, chosen


def run_all(feats: dict, y: np.ndarray, fold: np.ndarray, group: np.ndarray, pipelines: list) -> dict:
    """Out-of-fold scores and chosen settings for every pipeline in the list."""
    return {p: run_pipeline(feats[p[0]], y, fold, group, p[1]) for p in pipelines}


# --------------------------------------------------------------------------
# 4. Metrics and the permutation test
# --------------------------------------------------------------------------
def score_pipelines(out: dict, y: np.ndarray, group: np.ndarray) -> dict:
    """Within-row AUC, 95% bootstrap CI and paired Δ vs the baseline for every pipeline.

    :param out: {pipeline: (scores (n,), chosen settings)} from :func:`run_all`.
    :type out: dict
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param group: fold × row id per leaf, length n.
    :type group: np.ndarray
    :return: {pipeline: {"auc", "lo", "hi", "d", "d_lo", "d_hi", "p_better", "pooled_auc", "chosen"}}.
    :rtype: dict
    """
    from sklearn.metrics import roc_auc_score  # standard metric code

    w = bootstrap_weights(y, group, N_BOOT, BOOT_SEED)
    base_s = out[BASELINE][0]
    base_auc, base_boot = within_group_auc(base_s, y, group), within_group_auc(base_s, y, group, w)
    res = {}
    for p, (s, chosen) in out.items():
        auc, boot = within_group_auc(s, y, group), within_group_auc(s, y, group, w)
        diff = boot - base_boot
        res[p] = {"auc": auc, "lo": torch.quantile(boot, 0.025).item(), "hi": torch.quantile(boot, 0.975).item(),
                  "d": auc - base_auc, "d_lo": torch.quantile(diff, 0.025).item(),
                  "d_hi": torch.quantile(diff, 0.975).item(), "p_better": (diff > 0).double().mean().item(),
                  "pooled_auc": float(roc_auc_score(y, s)), "chosen": [str(c) for c in chosen]}
    return res


def _one_permutation(seed: int, feats: dict, y: np.ndarray, rows: np.ndarray, fold: np.ndarray,
                     group: np.ndarray) -> dict:
    """Shuffle labels within each row and rerun every linear pipeline (inner CV included)."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for r in np.unique(rows):
        m = np.flatnonzero(rows == r)
        yp[m] = y[rng.permutation(m)]
    out = run_all(feats, yp, fold, group, LINEAR)
    return {p: within_group_auc(s, yp, group) for p, (s, _c) in out.items()}


def permutation_test(feats: dict, y: np.ndarray, rows: np.ndarray, fold: np.ndarray, group: np.ndarray,
                     observed: dict, n_perm: int, n_jobs: int) -> dict:
    """Null distribution of within-row AUC for every linear pipeline, labels shuffled within rows.

    Shuffling within a row keeps each row's disease rate, so the null keeps whatever the
    row alone explains. p for the primary pipeline uses its own null; p for the best
    linear pipeline uses the null of the best of all linear pipelines in each permutation
    (max statistic), which accounts for having picked it from many.

    :param feats: Feature sets.
    :type feats: dict
    :param y: 0/1 labels, length n.
    :type y: np.ndarray
    :param rows: Row per leaf, length n.
    :type rows: np.ndarray
    :param fold: Fold per leaf, length n.
    :type fold: np.ndarray
    :param group: fold × row id per leaf, length n.
    :type group: np.ndarray
    :param observed: {pipeline: observed within-row AUC}.
    :type observed: dict
    :param n_perm: Number of permutations.
    :type n_perm: int
    :param n_jobs: Parallel workers.
    :type n_jobs: int
    :return: p-values, the best linear pipeline, and null summaries.
    :rtype: dict
    """
    from joblib import Parallel, delayed  # why: CPU-parallel loop over independent permutations

    null = Parallel(n_jobs=n_jobs)(delayed(_one_permutation)(PERM_SEED * 100000 + i, feats, y, rows, fold, group)
                                   for i in range(n_perm))
    prim = np.array([n[PRIMARY] for n in null])
    mx = np.array([max(n.values()) for n in null])
    best = max(LINEAR, key=lambda p: observed[p])
    return {
        "n_perm": n_perm,
        "primary": pname(PRIMARY), "primary_auc": observed[PRIMARY],
        "primary_p": float((1 + (prim >= observed[PRIMARY]).sum()) / (1 + n_perm)),
        "primary_null_95": float(np.quantile(prim, 0.95)),
        "best_linear": pname(best), "best_linear_auc": observed[best],
        "best_linear_p_familywise": float((1 + (mx >= observed[best]).sum()) / (1 + n_perm)),
        "max_null_95": float(np.quantile(mx, 0.95)),
        "baseline_p": float((1 + (np.array([n[BASELINE] for n in null]) >= observed[BASELINE]).sum())
                            / (1 + n_perm)),
    }


def univariate(feats: dict, y: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Each C/D feature and leaf size on its own: within-row AUC with 95% CI (no model, no folds).

    AUC > 0.5 means the feature is higher on red-blotch leaves.
    """
    group = np.array([ROWS.index(r) for r in rows])
    w = bootstrap_weights(y, group, N_BOOT, BOOT_SEED)
    cols = ([(f"{n} (leaf mean)", feats["C"][:, i]) for i, n in enumerate(feats["names_C"])]
            + [(n, feats["D"][:, i]) for i, n in enumerate(feats["names_D"])]
            + [("leaf size (pixels)", feats["n_px"]), ("interior pixels", feats["n_interior"])])
    out = []
    for name, v in cols:
        boot = within_group_auc(v, y, group, w)
        out.append({"feature": name, "auc": within_group_auc(v, y, group),
                    "lo": torch.quantile(boot, 0.025).item(), "hi": torch.quantile(boot, 0.975).item()})
    df = pd.DataFrame(out)
    return df.reindex((df.auc - 0.5).abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Report and figures
# --------------------------------------------------------------------------
def to_markdown(metrics: dict, size_auc: dict, perm: dict, uni: pd.DataFrame, names_E: list) -> str:
    """All result tables as Markdown; names_E names the columns of feature set E."""
    L = [f"# Red blotch from leaf spectra, {VARIETY}: results", "",
         "Within-row AUC: leaves of the same fold and row only (0.5 = chance), 95% bootstrap CI. "
         "Δ: paired difference to the baseline (A · logistic, C = 1) on the same resamples.", "",
         f"## Pre-declared primary: {pname(PRIMARY)} vs baseline, `{PERM_SCHEME}` folds", "",
         f"- within-row AUC {metrics[PERM_SCHEME][PRIMARY]['auc']:.3f} "
         f"[{metrics[PERM_SCHEME][PRIMARY]['lo']:.3f}, {metrics[PERM_SCHEME][PRIMARY]['hi']:.3f}]; "
         f"Δ vs baseline {metrics[PERM_SCHEME][PRIMARY]['d']:+.3f} "
         f"[{metrics[PERM_SCHEME][PRIMARY]['d_lo']:+.3f}, {metrics[PERM_SCHEME][PRIMARY]['d_hi']:+.3f}]",
         f"- permutation p (labels shuffled within rows, whole pipeline rerun, {perm['n_perm']} permutations): "
         f"**{perm['primary_p']:.3f}** (null 95th percentile {perm['primary_null_95']:.3f}); "
         f"baseline p = {perm['baseline_p']:.3f}",
         f"- exploratory: best linear pipeline under `{PERM_SCHEME}` = {perm['best_linear']} "
         f"(AUC {perm['best_linear_auc']:.3f}), family-wise p = {perm['best_linear_p_familywise']:.3f} "
         f"(max-statistic null 95th percentile {perm['max_null_95']:.3f})", "",
         "## Leaf size alone (pixel count as the score)", "",
         "| scheme | within-row AUC |", "|---|---|"]
    L += [f"| `{s}` | {a:.3f} |" for s, a in size_auc.items()]
    for s in SCHEMES:
        L += ["", f"## `{s}` folds: {SCHEMES[s]}", "",
              "| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |",
              "|---|---|---|---|---|---|"]
        for p in PIPELINES:
            m = metrics[s][p]
            d = "—" if p == BASELINE else f"{m['d']:+.3f} [{m['d_lo']:+.3f}, {m['d_hi']:+.3f}]"
            pb = "—" if p == BASELINE else f"{m['p_better']:.2f}"
            chosen = m["chosen"]
            if p[1] == "single":
                chosen = [c[0] + names_E[int(c[1:])] for c in chosen]
            L.append(f"| {pname(p)} | {m['auc']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}] | {d} | {pb} | "
                     f"{m['pooled_auc']:.3f} | {', '.join(chosen)} |")
    L += ["", "## Single features (no model), within-row AUC, top 20", "",
          "AUC > 0.5: higher on red-blotch leaves. 57 features screened, so expect ~3 with a CI excluding 0.5 by chance.", "",
          "| feature | AUC [95% CI] |", "|---|---|"]
    L += [f"| {r.feature} | {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}] |" for r in uni.head(20).itertuples()]
    return "\n".join(L) + "\n"


def fig_pipelines(metrics: dict, path: Path) -> None:
    """Dot plot: every pipeline's within-row AUC with 95% CI, one marker per fold scheme."""
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    n = len(PIPELINES)
    for i, p in enumerate(PIPELINES):
        for j, s in enumerate(SCHEMES):
            m = metrics[s][p]
            y = n - 1 - i + (1 - j) * 0.22
            ax.plot([m["lo"], m["hi"]], [y, y], color=SCHEME_COLORS[s], lw=1.6, solid_capstyle="round")
            ax.plot(m["auc"], y, "o", color=SCHEME_COLORS[s], ms=6, mec="white", mew=1.2, zorder=3,
                    label=s if i == 0 else None)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(range(n))
    labels = [pname(p) for p in PIPELINES][::-1]
    ax.set_yticklabels(labels, fontsize=9)
    for t in ax.get_yticklabels():
        if t.get_text() == pname(PRIMARY):
            t.set_fontweight("bold")
    ax.set_xlabel("within-row ROC AUC, red blotch (Chardonnay)", fontsize=10, color=INK_2)
    ax.set_xlim(0.3, 0.85)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.legend(title="folds", fontsize=9, title_fontsize=9, frameon=False, loc="lower right")
    fig.text(0.99, 0.01, "dot = within-row AUC with 95% bootstrap CI; bold = pre-declared primary; dashed = chance",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_features(uni: pd.DataFrame, path: Path, top: int = 20) -> None:
    """Top single features by distance from 0.5, with 95% CI; leaf size always shown."""
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    d = uni.head(top)
    if "leaf size (pixels)" not in set(d.feature):
        d = pd.concat([d, uni[uni.feature == "leaf size (pixels)"]])
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(d) + 1.2))
    for i, r in enumerate(d.itertuples()):
        y = len(d) - 1 - i
        color = MUTED if "size" in r.feature or "interior" in r.feature else "#2a78d6"
        ax.plot([r.lo, r.hi], [y, y], color=color, lw=1.8, solid_capstyle="round")
        ax.plot(r.auc, y, "o", color=color, ms=6, mec="white", mew=1.2, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.feature)[::-1], fontsize=8.5)
    ax.set_xlabel("within-row AUC alone (> 0.5: higher on red-blotch leaves)", fontsize=9.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Load → features → every pipeline under every scheme → permutation test → report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000, help="permutations for the p-values")
    ap.add_argument("--jobs", type=int, default=12, help="parallel workers for the permutation test")
    args = ap.parse_args()
    for d in ("results", "figures", "logs"):
        (HERE / d).mkdir(exist_ok=True)
    log = get_logger("red_blotch_spectra", log_file=HERE / "logs" / "run.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    full = load_meta()
    meta = full[full.variety == VARIETY].reset_index(drop=True)
    y, rows = meta.red_blotch.to_numpy(), meta.row.to_numpy()
    feats = load_features(meta, load_wavelengths(), device)
    log.info("%s: %d leaves, %d red blotch | features A %d, B %d, C %d, D %d, E %d | %.0f s",
             VARIETY, len(y), y.sum(), *(feats[k].shape[1] for k in "ABCDE"), time.time() - t0)

    metrics, size_auc, folds = {}, {}, {}
    for s in SCHEMES:
        fold = fold_of(meta, s)
        group = fold * 5 + np.array([ROWS.index(r) for r in rows])
        folds[s] = (fold, group)
        out = run_all(feats, y, fold, group, PIPELINES)
        metrics[s] = score_pipelines(out, y, group)
        size_auc[s] = within_group_auc(feats["n_px"], y, group)
        pd.DataFrame({"path": meta.path, "fold": fold, "red_blotch": y,
                      **{pname(p): sc for p, (sc, _c) in out.items()}}).to_csv(HERE / "results" / f"oof_{s}.csv",
                                                                               index=False)
        for p in PIPELINES:
            m = metrics[s][p]
            log.info("%-5s | %-48s | AUC %.3f [%.3f, %.3f] | Δ %+.3f [%+.3f, %+.3f]", s, pname(p), m["auc"],
                     m["lo"], m["hi"], m["d"], m["d_lo"], m["d_hi"])
        log.info("%-5s | leaf size alone: AUC %.3f | %.0f s", s, size_auc[s], time.time() - t0)

    fold, group = folds[PERM_SCHEME]
    perm = permutation_test(feats, y, rows, fold, group, {p: metrics[PERM_SCHEME][p]["auc"] for p in LINEAR},
                            args.n_perm, args.jobs)
    log.info("[permutation, %s, %d] primary %s: p = %.3f | best linear %s: family-wise p = %.3f | %.0f s",
             PERM_SCHEME, perm["n_perm"], perm["primary"], perm["primary_p"], perm["best_linear"],
             perm["best_linear_p_familywise"], time.time() - t0)

    uni = univariate(feats, y, rows)
    uni.to_csv(HERE / "results" / "univariate.csv", index=False)
    names_E = [f"{n} (leaf mean)" for n in feats["names_C"]] + list(feats["names_D"])
    (HERE / "results" / "metrics.md").write_text(to_markdown(metrics, size_auc, perm, uni, names_E))
    (HERE / "results" / "metrics.json").write_text(json.dumps(
        {"metrics": {s: {pname(p): m for p, m in ms.items()} for s, ms in metrics.items()},
         "leaf_size_auc": size_auc, "permutation": perm}, indent=2))
    fig_pipelines(metrics, HERE / "figures" / "auc_by_pipeline.png")
    fig_features(uni, HERE / "figures" / "feature_auc.png")
    log.info("[saved] results/metrics.md, results/univariate.csv, figures/*.png | total %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
