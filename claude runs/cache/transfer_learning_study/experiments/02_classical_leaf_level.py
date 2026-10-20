"""APPROACH 1 - classical models on ONE summary spectrum per leaf.

Run:   python experiments/02_classical_leaf_level.py          (full, ~3-5 min on 14 cores)
       python experiments/02_classical_leaf_level.py --quick  (fold 0 only, ~1 min)
       python experiments/02_classical_leaf_level.py --lovo   (leave-one-variety-out)
       python experiments/02_classical_leaf_level.py --by-variety  (separate model per variety)

Research basis: reports/Hyperspectral grapevine transfer learning.md - the
"bar to beat" for small hyperspectral datasets is a tuned SVM / logistic
regression / boosting model on processed spectra, not a deep network.
(Tobacco-leaf study, 95 leaves: SVM 97.9%, XGBoost 97.5%, transformer 99.35%.)

WHAT THIS DOES
  Each leaf (thousands of pixels) is squeezed into a single row of numbers:
      mean spectrum          204 values   (average over leaf pixels, per band)
      "stats"                816 values   (mean + std + 10th + 90th percentile)
  then optionally pre-processed:
      raw    : as is
      snv    : Standard Normal Variate - each spectrum is centred and scaled
               by its own mean/std; removes overall brightness differences
      deriv  : Savitzky-Golay 1st derivative; removes baseline offsets and
               emphasises the SHAPE of the curve (e.g. the red edge)
  and fed to four models, each trained separately for each disease:
      logreg : logistic regression (linear)
      svm    : support vector machine with RBF kernel (non-linear)
      plsda  : partial least squares regression on the 0/1 label - the
               classic chemometrics method for correlated bands
      hgb    : histogram gradient boosting (trees)

NO TORCH HERE - this is plain scikit-learn. Every model is wrapped in a
Pipeline([StandardScaler, model]); a Pipeline re-fits the scaler on the
TRAINING leaves of each fold only, so nothing leaks from the test fold.
No hyper-parameter tuning on the test folds: fixed, sensible defaults.
"""
import os

# Speed: we run many small model fits in PARALLEL processes (one per fold x setting),
# so each fit should use only 2 CPU threads. Must be set BEFORE sklearn/numpy load.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (CACHE_DIR, LABELS, RESULTS_DIR, VARIETIES, fold_indices, n_folds, label_mask,
                    leaf_pixels, load_meta, load_pixels, print_fold_summary,
                    save_oof)


# --------------------------------------------------------------------------
# 1. Turn every leaf into summary statistics (cached after the first run)
# --------------------------------------------------------------------------
def leaf_statistics(meta, pixels):
    """(N, 4, 204) array: mean, std, p10, p90 of each band over leaf pixels."""
    cache = CACHE_DIR / "leaf_stats.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros((len(meta), 4, pixels.shape[1]), dtype=np.float32)
    for i in range(len(meta)):
        px = leaf_pixels(pixels, meta, i)
        out[i, 0] = px.mean(0)
        out[i, 1] = px.std(0)
        out[i, 2:] = np.percentile(px, [10, 90], axis=0)
    np.save(cache, out)
    return out


def snv(x):
    """Standard Normal Variate: each row (spectrum) -> zero mean, unit std."""
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


def build_features(stats, name):
    mean = stats[:, 0]
    if name == "mean_raw":
        return mean
    if name == "mean_snv":
        return snv(mean)
    if name == "mean_deriv":      # window 11 bands (~35 nm), 2nd-order polynomial
        return savgol_filter(mean, window_length=11, polyorder=2, deriv=1, axis=1)
    if name == "stats_raw":
        return stats.reshape(len(stats), -1)
    raise ValueError(name)


# --------------------------------------------------------------------------
# 2. Models. All expose fit(X, y) and predict_proba(X)[:, 1].
# --------------------------------------------------------------------------
class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS Discriminant Analysis = PLS regression onto the 0/1 label.

    The regression output is a continuous score; we clip it into [0, 1] and
    treat it as a probability-like value (fine for AUC, which only needs ranks).
    """

    def __init__(self, n_components=8):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, y)
        return self

    def predict_proba(self, X):
        s = np.clip(self.pls_.predict(X).ravel(), 1e-3, 1 - 1e-3)
        return np.c_[1 - s, s]


class SVMScore(BaseEstimator, ClassifierMixin):
    """RBF-SVM whose distance-to-the-boundary is squashed into (0, 1).

    (sklearn's built-in SVC(probability=True) is deprecated; for AUC only the
    ranking of the scores matters, and a sigmoid keeps that ranking.)
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.svc_ = SVC(kernel="rbf", C=self.C, gamma="scale",
                        class_weight="balanced").fit(X, y)
        return self

    def predict_proba(self, X):
        s = 1.0 / (1.0 + np.exp(-self.svc_.decision_function(X)))
        return np.c_[1 - s, s]


def make_model(name):
    scaler = StandardScaler()
    if name == "logreg":
        # C=0.1 = fairly strong L2 regularisation (many correlated features, few leaves)
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    elif name == "svm":
        m = SVMScore(C=1.0)
    elif name == "plsda":
        m = PLSDA(n_components=8)
    elif name == "hgb":
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                           max_depth=3, class_weight="balanced",
                                           random_state=0)
    else:
        raise ValueError(name)
    return make_pipeline(scaler, m)


# --------------------------------------------------------------------------
# 3. Cross-validation
# --------------------------------------------------------------------------
def fit_predict_fold(meta, X, y, mask, model_name, train_idx, test_idx, by_variety):
    """Train on train_idx, return (len(test_idx), 2) probabilities [red_blotch, leafroll3].

    by_variety=False : one model for all varieties.
    by_variety=True  : a separate model for each variety, trained only on that
                       variety's training leaves. Motivation (see 10_diagnostics.py):
                       the direction of the spectral signal differs between varieties,
                       so one pooled model cannot use it. If a variety has fewer than
                       10 leaves of either class in the training folds (Cab Franc has
                       almost no healthy leaves), we fall back to its training prevalence.
    """
    out = np.zeros((len(test_idx), 2))
    groups = [(None, train_idx, np.arange(len(test_idx)))]
    if by_variety:
        v_all = meta.variety.to_numpy()
        groups = [(v, train_idx[v_all[train_idx] == v], np.where(v_all[test_idx] == v)[0])
                  for v in VARIETIES]
    for v, tr_all, te_pos in groups:
        if len(te_pos) == 0:
            continue
        for j in range(2):                       # j=0 red_blotch, j=1 leafroll3
            tr = tr_all[mask[tr_all, j]]         # leafroll: drop Cab Franc
            yj = y[tr, j]
            if by_variety and (len(tr) == 0 or min((yj == 0).sum(), (yj == 1).sum()) < 10):
                out[te_pos, j] = yj.mean() if len(tr) else 0.5
                continue
            model = clone(make_model(model_name)).fit(X[tr], yj)
            out[te_pos, j] = model.predict_proba(X[test_idx[te_pos]])[:, 1]
    return out


def run_all(meta, stats, feats, models, folds, quick, scheme, skip_existing, by_variety, n_jobs):
    """Run every (feature set x model) combination, all folds, in parallel processes.

    joblib.Parallel starts n_jobs worker processes and hands each task
    (one fold of one combination) to whichever worker is free. On a 14-core
    machine this is ~7x faster than looping over folds one at a time.
    """
    y = meta[LABELS].to_numpy()
    mask = label_mask(meta)
    X_by_feat = {f: build_features(stats, f) for f in feats}

    combos = []                                   # (feature, model, run name)
    for feat in feats:
        for model_name in models:
            name = (f"02_{model_name}__{feat}" + ("__lovo" if scheme == "lovo" else "")
                    + ("__byvariety" if by_variety else ""))
            done = RESULTS_DIR / ("quick" if quick else "") / f"{name}_oof.csv"
            if skip_existing and done.exists():
                print(f"[{name}] already done, skipping")
                continue
            combos.append((feat, model_name, name))

    tasks = [(ci, fold) for ci in range(len(combos)) for fold in folds]
    t0 = time.time()
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(fit_predict_fold)(meta, X_by_feat[combos[ci][0]], y, mask, combos[ci][1],
                                  *fold_indices(meta, fold, scheme), by_variety)
        for ci, fold in tasks)
    print(f"all fits finished in {time.time() - t0:.0f}s")

    preds = {ci: np.full((len(meta), 2), np.nan) for ci in range(len(combos))}
    for (ci, fold), out in zip(tasks, outs):
        _, test_idx = fold_indices(meta, fold, scheme)
        preds[ci][test_idx] = out
    for ci, (feat, model_name, name) in enumerate(combos):
        for fold in folds:
            _, test_idx = fold_indices(meta, fold, scheme)
            print_fold_summary(name, fold, meta, test_idx, preds[ci][test_idx, 0],
                               preds[ci][test_idx, 1])
        save_oof(name, meta, preds[ci][:, 0], preds[ci][:, 1], quick=quick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fold 0 only; saves to results/quick/")
    ap.add_argument("--lovo", action="store_true",
                    help="leave-one-variety-out instead of the 5 row folds (stress test)")
    ap.add_argument("--by-variety", action="store_true",
                    help="train a separate model inside each variety")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip combinations whose results file already exists (to resume)")
    ap.add_argument("--models", default="logreg,svm,plsda,hgb")
    ap.add_argument("--features", default="mean_raw,mean_snv,mean_deriv,stats_raw")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="parallel worker processes (default: half the CPU cores)")
    args = ap.parse_args()

    meta = load_meta()
    stats = leaf_statistics(meta, load_pixels(mmap=True))
    scheme = "lovo" if args.lovo else "row"
    folds = [0] if args.quick else list(range(n_folds(scheme)))
    run_all(meta, stats, args.features.split(","), args.models.split(","), folds, args.quick,
            scheme, args.skip_existing, args.by_variety, args.jobs)


if __name__ == "__main__":
    main()
