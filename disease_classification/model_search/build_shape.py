"""Leaf shape and edge-to-centre profile features, aimed at leafroll's downward-rolled leaf margins.

Downward rolling of the margins is the leafroll symptom shared by red and white cultivars; red blotch
usually does not roll leaves (Frontiers in Microbiology 2013 leafroll reviews; OMAFRA leafroll fact sheet).
Seen from above, a rolled leaf has a shorter, less convex outline and darker, foreshortened margins.
No paper found that uses these on hyperspectral leaf images; this is the search's own design.

Per leaf (from its mask, background = 0 in every band):
- shape (17): log area, log perimeter, perimeter^2 / area, solidity, extent, eccentricity, minor/major axis,
  log convex area - log area, the 7 Hu (1962) moment invariants (signed log), and 2 margin-band shares
  (pixels within 3 / 8 px of the edge).
- profile (30): mean reflectance at 550, 670 and 800 nm in 10 bins of normalised distance to the edge
  (distance / the leaf's largest distance), each divided by that band's leaf mean.
- zone (28): SNV spectrum of the outer zone (normalised distance < 0.2) minus that of the inner zone
  (> 0.5), averaged in 20 nm bins over 420-980 nm.
Saved to cache/disease_search/morph.npz in load_meta("all") order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_cubes, load_dataset  # noqa: E402
from logistic_regression import load_meta  # noqa: E402


def snv(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / s.std()


def leaf_features(c: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shape, profile, zone) features of one cube (204, lines, samples)."""
    mask = (c != 0).any(0)
    lab = label(mask)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1          # largest component = the leaf
    mask = lab == big
    rp = regionprops(mask.astype(np.uint8))[0]
    hu = rp.moments_hu
    d = distance_transform_edt(mask)
    shape = np.array([
        np.log(rp.area), np.log(rp.perimeter), rp.perimeter ** 2 / rp.area, rp.solidity, rp.extent,
        rp.eccentricity, rp.axis_minor_length / rp.axis_major_length, np.log(rp.area_convex) - np.log(rp.area),
        *(-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)),
        ((d > 0) & (d <= 3)).sum() / rp.area, ((d > 0) & (d <= 8)).sum() / rp.area])
    dn = d / d.max()
    bins = np.minimum((dn[mask] * 10).astype(int), 9)
    prof = []
    for nm in (550, 670, 800):
        b = c[int(np.abs(w - nm).argmin())][mask].astype(np.float64)
        prof += list(np.bincount(bins, weights=b, minlength=10) / np.maximum(np.bincount(bins, minlength=10), 1) / b.mean())
    outer = c[:, mask & (dn < 0.2)].astype(np.float64).mean(1)
    inner = c[:, mask & (dn > 0.5)].astype(np.float64).mean(1)
    diff = snv(outer) - snv(inner)
    edges = np.arange(420, 981, 20)
    zone = np.array([diff[(w >= lo) & (w < lo + 20)].mean() for lo in edges[:-1]])
    return shape, np.array(prof), zone


def main() -> None:
    meta = load_meta("all")
    w = load_dataset()[1]
    S, P, Z, paths = [], [], [], []
    for v in meta.variety.unique():
        m, cubes = load_cubes(v)
        paths += list(m.path)
        for c in cubes:
            s, p, z = leaf_features(c, w)
            S.append(s), P.append(p), Z.append(z)
        del cubes
    assert paths == list(meta.path)
    S, P, Z = np.array(S), np.array(P), np.array(Z)
    assert np.isfinite(S).all() and np.isfinite(P).all() and np.isfinite(Z).all()
    np.savez(ROOT / "cache/disease_search/morph.npz", shape=S, profile=P, zone=Z, paths=np.array(paths))
    print("[saved] morph.npz", S.shape, P.shape, Z.shape, flush=True)


if __name__ == "__main__":
    main()
