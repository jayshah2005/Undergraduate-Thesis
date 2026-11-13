"""Sample 400 interior pixels per leaf (3-px margin eroded, as LeafPool ignores the margin), pixel SNV over
204 bands, saved float16 to cache/disease_search/pixels.npz in load_meta("all") order (seed 0)."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_cubes  # noqa: E402
from logistic_regression import load_meta  # noqa: E402
NPIX = 400


def main() -> None:
    """Sample every leaf's pixels (seed 0) and save them."""
    meta = load_meta("all")
    rng = np.random.default_rng(0)
    out, paths = [], []
    for v in meta.variety.unique():
        m, cubes = load_cubes(v)
        paths += list(m.path)
        for c in cubes:
            mask = (c != 0).any(0)
            inner = binary_erosion(mask, iterations=3)
            ys, xs = np.nonzero(inner if inner.sum() >= NPIX else mask)
            pick = rng.choice(len(ys), NPIX, replace=len(ys) < NPIX)
            R = c[:, ys[pick], xs[pick]].T.astype(np.float64)            # (NPIX, 204)
            Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
            out.append(Z.astype(np.float16))
        del cubes
    assert paths == list(meta.path)
    P = np.stack(out)
    assert np.isfinite(P.astype(np.float32)).all()
    np.savez(ROOT / "cache/disease_search/pixels.npz", P=P, paths=np.array(paths))
    print("[saved] pixels.npz", P.shape)


if __name__ == "__main__":
    main()
