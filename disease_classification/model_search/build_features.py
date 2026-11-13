"""Per-leaf feature sets for the disease model search, cached in cache/disease_search/features.npz.

Every array has one row per leaf in load_meta("all") order (variety, row, vine).

- mean_refl (204): leaf-mean reflectance (must equal helpers.data's cached means).
- snv_q (3, 204): per-band 10th / 50th / 90th percentile of the pixel-SNV spectra (patchy symptoms
  are washed out by a leaf mean; quantiles keep them).
- snv_sd (204): per-band SD of the pixel-SNV spectra.
- idx_stats (n_idx, 7): per-pixel vegetation indices, summarised per leaf by mean, SD and the
  5/25/50/75/95th percentiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_cubes, load_dataset  # noqa: E402
from logistic_regression import load_meta  # noqa: E402

OUT = ROOT / "cache/disease_search/features.npz"
QS = (0.10, 0.50, 0.90)
STAT_Q = (0.05, 0.25, 0.50, 0.75, 0.95)


def band(w: np.ndarray, nm: float) -> int:
    """Index of the band nearest nm."""
    return int(np.abs(w - nm).argmin())


def indices(R: torch.Tensor, w: np.ndarray) -> dict[str, torch.Tensor]:
    """Per-pixel vegetation indices from reflectance R (n_pix, 204); nearest band to each wavelength."""
    r = lambda nm: R[:, band(w, nm)].clamp_min(1e-4)  # noqa: E731
    out = {
        "NDVI": (r(800) - r(670)) / (r(800) + r(670)),
        "GNDVI": (r(800) - r(550)) / (r(800) + r(550)),
        "NDRE": (r(790) - r(720)) / (r(790) + r(720)),
        "ARI": 1 / r(550) - 1 / r(700),
        "mARI": (1 / r(550) - 1 / r(700)) * r(800),
        "CRI1": 1 / r(510) - 1 / r(550),
        "CRI2": 1 / r(510) - 1 / r(700),
        "PRI": (r(531) - r(570)) / (r(531) + r(570)),
        "CI_rededge": r(780) / r(710) - 1,
        "CI_green": r(780) / r(550) - 1,
        "REP": 700 + 40 * ((r(670) + r(780)) / 2 - r(700)) / (r(740) - r(700)).clamp_min(1e-4),
        "MCARI": ((r(700) - r(670)) - 0.2 * (r(700) - r(550))) * (r(700) / r(670)),
        "PSRI": (r(680) - r(500)) / r(750),
        "SIPI": (r(800) - r(445)) / (r(800) - r(680)).clamp_min(1e-4),
        "VOG1": r(740) / r(720),
    }
    rgb = torch.stack([r(640), r(550), r(460)], 1)
    chrom = rgb / rgb.sum(1, keepdim=True)
    out.update({"chrom_r": chrom[:, 0], "chrom_g": chrom[:, 1], "chrom_b": chrom[:, 2]})
    return out


def leaf_features(cube: np.ndarray, w: np.ndarray, dev: torch.device) -> dict[str, np.ndarray]:
    """Feature sets of one leaf cube (204, lines, samples), background exactly 0."""
    c = torch.from_numpy(cube).to(dev)
    mask = (c != 0).any(0)
    R = c[:, mask].T.double()                           # (n_pix, 204) reflectance
    mu, sd = R.mean(1, keepdim=True), R.std(1, keepdim=True, unbiased=False).clamp_min(1e-12)
    Z = (R - mu) / sd                                   # pixel SNV
    q = torch.tensor(QS, dtype=Z.dtype, device=dev)
    sq = torch.tensor(STAT_Q, dtype=Z.dtype, device=dev)
    ind = indices(R, w)
    stats = []
    for v in ind.values():
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        stats.append(torch.cat([v.mean()[None], v.std(unbiased=False)[None], torch.quantile(v, sq)]))
    return {"mean_refl": R.mean(0).cpu().numpy(), "snv_q": torch.quantile(Z, q, dim=0).cpu().numpy(),
            "snv_sd": Z.std(0, unbiased=False).cpu().numpy(), "idx_stats": torch.stack(stats).cpu().numpy(),
            "n_pix": np.array(R.shape[0])}


def main() -> None:
    """Load every cube once, compute its features, check the means, save."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_all, w, means, _g = load_dataset()
    meta = load_meta("all")
    feats: dict[str, list] = {}
    paths = []
    for v in meta.variety.unique():
        m, cubes = load_cubes(v)
        paths += list(m.path)
        for cube in cubes:
            for k, a in leaf_features(cube, w, dev).items():
                feats.setdefault(k, []).append(a)
        del cubes
    assert paths == list(meta.path), "cube order differs from load_meta"
    out = {k: np.stack(a) for k, a in feats.items()}
    # The cached leaf means were computed in float64 from the same pixels.
    assert np.allclose(out["mean_refl"], means, rtol=1e-5, atol=1e-7), "leaf means differ from helpers.data"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out, paths=np.array(paths), wavelengths=w,
             index_names=np.array(list(indices(torch.ones(1, 204, dtype=torch.float64), w))))
    print(f"[saved] {OUT} " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
