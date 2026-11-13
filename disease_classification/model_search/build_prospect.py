"""Leaf pigment traits by inverting PROSPECT-D on each leaf's mean reflectance (needs `pip install prosail`).

PROSPECT-D (Féret, Gitelson, Noble & Jacquemoud 2017, "PROSPECT-D: Towards modeling leaf optical properties
through a complete lifecycle", Remote Sensing of Environment 193:204-215) models leaf reflectance from the leaf
structure N and the contents of chlorophyll a+b (Cab), carotenoids (Car), anthocyanins (Ant), brown pigments
(Cbrown), water (Cw) and dry matter (Cm). Retrieving plant traits by inverting such a model is how Zarco-Tejada
et al. 2018 (Nature Plants 4:432-439) found pre-visual Xylella infection; they also had fluorescence and thermal
traits, which this data does not.

The forward model is prosail's run_prospect (J. Gómez-Dans, https://github.com/jgomezdans/prosail, GPLv3),
called as a library; its output (400-2500 nm, 1 nm steps) is interpolated to the Specim IQ band centres.

Adaptations for image-based leaf reflectance (no integrating sphere, curved leaves, some glint), this repo's own:
- observed = s * PROSPECT(theta) + o, with the scale s and offset o solved by least squares for every theta
  (variable projection), so illumination level and specular offset do not bias the pigments. The pigments are
  therefore relative, PROSPECT-fitted values, not ug/cm2;
- fitted on 450-900 nm (noisy sensor ends left out), Cw fixed at 0.012 cm (water barely absorbs below 900 nm);
- bounds N 1-3, Cab 0-100, Car 0-30, Ant 0-10, Cbrown 0-1, Cm 0.001-0.03; best of 4 starting points.
N and Cm are poorly determined without short-wave infrared or transmittance; the pigments are well determined.

Output: cache/disease_search/prospect.npz in load_meta("all") order: traits (n, 11) = N, Cab, Car, Ant, Cbrown,
Cm, Car/Cab, Ant/Cab, s, o, fit RMSE; and the fit R^2 per leaf.
Note: the prospect.npz behind final_report.md was made with a line-by-line port of the same function that read
prosail's coefficient table (the package could not be installed in that session); it interpolated the
coefficients, not the reflectance, to the band centres. Rerunning this script may move the traits slightly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_dataset  # noqa: E402
from logistic_regression import load_meta  # noqa: E402

CW = 0.012
LO, HI = 450.0, 900.0
NAMES = ("N", "Cab", "Car", "Ant", "Cbrown", "Cm")
BOUNDS = (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.001]), np.array([3.0, 100.0, 30.0, 10.0, 1.0, 0.03]))
STARTS = [np.array(s) for s in ([1.5, 40, 8, 1, 0.05, 0.008], [2.0, 20, 5, 0.2, 0.0, 0.005],
                                 [1.3, 60, 12, 3, 0.2, 0.012], [2.5, 30, 10, 0.5, 0.5, 0.02])]
X_SCALE = np.array([0.5, 20, 5, 1, 0.2, 0.005])


def prospect(w: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """PROSPECT-D reflectance at the wavelengths w (nm) for theta = (N, Cab, Car, Ant, Cbrown, Cm)."""
    import prosail
    N, cab, car, ant, cbrown, cm = theta
    wv, refl, _trans = prosail.run_prospect(N, cab, car, cbrown, CW, cm, ant=ant, prospect_version="D")
    return np.interp(w, wv, refl)


def fit_leaf(w: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Best (N, Cab, Car, Ant, Cbrown, Cm, s, o), RMSE and R^2 of observed = s * PROSPECT + o on the bands w.

    :param w: Band centres in nm, length b.
    :param R: Leaf-mean reflectance on those bands, length b.
    """
    def scale_offset(theta):
        P = prospect(w, theta)
        so, *_ = np.linalg.lstsq(np.stack([P, np.ones_like(P)], 1), R, rcond=None)
        return P, so

    def resid(theta):
        P, (s, o) = scale_offset(theta)
        return R - (s * P + o)

    best = min((least_squares(resid, x0, bounds=BOUNDS, x_scale=X_SCALE, max_nfev=400) for x0 in STARTS),
               key=lambda f: f.cost)
    _, (s, o) = scale_offset(best.x)
    res = best.fun
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r2 = float(1 - (res ** 2).sum() / ((R - R.mean()) ** 2).sum())
    return np.concatenate([best.x, [s, o]]), rmse, r2


def main() -> None:
    """Fit every leaf mean, derive the ratios, save prospect.npz and print one summary line."""
    meta_all, w, means, _g = load_dataset()
    meta = load_meta("all")
    assert list(meta_all.path) == list(meta.path)
    band = (w >= LO) & (w <= HI)
    fits = [fit_leaf(w[band], R[band]) for R in means]
    X = np.array([f[0] for f in fits])
    rmse, r2 = np.array([f[1] for f in fits]), np.array([f[2] for f in fits])
    N, cab, car, ant, cbrown, cm, s, o = X.T
    traits = np.stack([N, cab, car, ant, cbrown, cm, car / np.maximum(cab, 1e-3), ant / np.maximum(cab, 1e-3),
                       s, o, rmse], 1)
    assert np.isfinite(traits).all()
    np.savez(ROOT / "cache/disease_search/prospect.npz", traits=traits, r2=r2, paths=meta.path.to_numpy(dtype=str),
             names=np.array([*NAMES, "Car/Cab", "Ant/Cab", "scale", "offset", "rmse"]))
    at_bound = (X[:, :6] <= BOUNDS[0] + 1e-6) | (X[:, :6] >= BOUNDS[1] - 1e-6)
    print(f"[saved] prospect.npz {traits.shape}; RMSE median {np.median(rmse):.4f}, R2 median {np.median(r2):.4f}; "
          "share at a bound: " + " ".join(f"{n} {a:.2f}" for n, a in zip(NAMES, at_bound.mean(0))))


if __name__ == "__main__":
    main()
