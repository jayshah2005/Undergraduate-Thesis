"""PCA math shared by the two questions under PCA/."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))
from data import load_dataset  # noqa: E402
from log import get_logger  # noqa: E402, F401

CLASSES = ["healthy", "RB only", "LR only", "both"]
CLASS_COLORS = ["#009E73", "#D55E00", "#56B4E9", "#6A3D9A"]
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
VAR_COLORS = {"Baco": "#E69F00", "Cab Franc": "#CC79A7", "Chardonnay": "#0072B2", "Vidal": "#555555"}
VAR_MARKERS = {"Baco": "o", "Cab Franc": "s", "Chardonnay": "^", "Vidal": "D"}
N_PCS, N_PERM, SEED = 10, 999, 0


def get_device() -> torch.device:
    """Pick the GPU if it is there; use it for every tensor in the run.

    :return: cuda if available, otherwise cpu.
    :rtype: torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_(t: torch.Tensor):
    """Copy a tensor to NumPy for matplotlib / JSON.

    Why NumPy: files and plots, not compute.

    :param t: Tensor on any device.
    :type t: torch.Tensor
    :return: A NumPy array on the CPU.
    """
    return t.detach().cpu().numpy()


def snv(X: torch.Tensor) -> torch.Tensor:
    """Centre and scale each spectrum over its own bands.

    For each vine, subtract the mean over its 204 bands and divide by that
    vine's SD. Removes overall brightness (sun, leaf angle, session) and
    keeps spectral shape. Population SD (unbiased=False): rescaling this
    curve, not estimating a population.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :return: SNV spectra of the same shape.
    :rtype: torch.Tensor
    """
    return (X - X.mean(1, keepdim=True)) / X.std(1, keepdim=True, unbiased=False)


def pca_fit(X: torch.Tensor, n_pcs: int = N_PCS):
    """Fit covariance PCA of X with a full SVD.

    998 x 204 is tiny, so this does not use a randomised approximation.
    Bands are not z-scored: noisy edge bands near 400/1000 nm would otherwise
    dominate.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :param n_pcs: Number of score columns to return.
    :type n_pcs: int
    :return: Scores Z (n, n_pcs), explained-variance ratio of those PCs,
        loadings (d, n_pcs), and the full explained-variance ratio.
    :rtype: tuple
    """
    # Centre each band by subtracting its mean over vines (axis 0), not over bands.
    Xc = X - X.mean(0)

    # Perform SVD (Singular Value Decomposition)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    # SVD signs are arbitrary. Flip each PC so its largest |loading| is positive.
    cols = torch.abs(Vh).argmax(1)
    signs = torch.sign(Vh[torch.arange(Vh.shape[0], device=Vh.device), cols])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    U = U * signs
    Vh = Vh * signs.unsqueeze(1)
    ev = (S ** 2) / (S ** 2).sum()
    Z = U[:, :n_pcs] * S[:n_pcs]
    return Z, ev[:n_pcs], Vh[:n_pcs].T, ev


def total_ss(X: torch.Tensor) -> torch.Tensor:
    """Sum squared distances of every row from the mean, over all columns.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :return: A scalar sum of squares.
    :rtype: torch.Tensor
    """
    return ((X - X.mean(0)) ** 2).sum()


def between_ss(X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Compute the between-group sum of squares.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :param g: Integer labels, length n.
    :type g: torch.Tensor
    :return: A scalar sum of squares.
    :rtype: torch.Tensor
    """
    gm = X.mean(0)
    ss = X.new_zeros(())
    for k in torch.unique(g):
        m = g == k
        ss = ss + m.sum() * ((X[m].mean(0) - gm) ** 2).sum()
    return ss


def between_ss_within(X: torch.Tensor, g: torch.Tensor, strata: torch.Tensor) -> torch.Tensor:
    """Sum between-group SS inside each stratum (group effect net of the strata).

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :param g: Integer labels, length n.
    :type g: torch.Tensor
    :param strata: Stratum ids, length n (e.g. variety).
    :type strata: torch.Tensor
    :return: A scalar sum of squares.
    :rtype: torch.Tensor
    """
    ss = X.new_zeros(())
    for s in torch.unique(strata):
        m = strata == s
        ss = ss + between_ss(X[m], g[m])
    return ss


def r2(X: torch.Tensor, g: torch.Tensor) -> float:
    """Return the share of total SS lying between the groups of g.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :param g: Integer labels, length n.
    :type g: torch.Tensor
    :return: 0 if the groups are identical, else between_ss / total_ss.
    :rtype: float
    """
    tot = total_ss(X)
    if tot.item() == 0:
        return 0.0
    return float(between_ss(X, g) / tot)


def permute_within(g: torch.Tensor, strata: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Shuffle labels separately inside each stratum, keeping that stratum's counts.

    Why within rows: disease rate differs by vineyard row and rows differ
    spectrally, so a free shuffle would let a row effect pass for disease.
    randperm uses a CPU generator (PyTorch requirement); indices are then
    moved onto g's device.

    :param g: Integer labels, length n.
    :type g: torch.Tensor
    :param strata: Stratum ids, length n.
    :type strata: torch.Tensor
    :param generator: Seeded CPU generator for randperm.
    :type generator: torch.Generator
    :return: A shuffled copy of g.
    :rtype: torch.Tensor
    """
    out = g.clone()
    for s in torch.unique(strata):
        idx = torch.nonzero(strata == s, as_tuple=False).flatten()
        perm = torch.randperm(idx.numel(), generator=generator)
        out[idx] = g[idx[perm.to(g.device)]]
    return out


def r2_test(X: torch.Tensor, g: torch.Tensor, strata: torch.Tensor | None,
            generator: torch.Generator, n_perm: int = N_PERM) -> tuple[float, float, float]:
    """Return observed R², permutation p-value, and mean null R².

    p = (1 + #shuffles with R² >= observed) / (1 + n_perm). strata=None permutes freely.

    :param X: Spectra, shape (n, d).
    :type X: torch.Tensor
    :param g: Integer labels, length n.
    :type g: torch.Tensor
    :param strata: Stratum ids, or None for a free shuffle.
    :type strata: torch.Tensor | None
    :param generator: Seeded CPU generator for randperm.
    :type generator: torch.Generator
    :param n_perm: Number of shuffles.
    :type n_perm: int
    :return: Observed R², p-value, mean null R².
    :rtype: tuple
    """
    obs = r2(X, g)
    n_ge, null_sum = 1, 0.0
    n = g.numel()
    for _ in range(n_perm):
        if strata is None:
            gp = g[torch.randperm(n, generator=generator).to(g.device)]
        else:
            gp = permute_within(g, strata, generator)
        v = r2(X, gp)
        null_sum += v
        if v >= obs:
            n_ge += 1
    return obs, n_ge / (1 + n_perm), null_sum / n_perm


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-band (mean b - mean a) / pooled SD.

    :param a: Reference group, shape (n_a, d).
    :type a: torch.Tensor
    :param b: Contrast group, shape (n_b, d).
    :type b: torch.Tensor
    :return: Effect size per band, shape (d,).
    :rtype: torch.Tensor
    """
    na, nb = a.shape[0], b.shape[0]
    sp = torch.sqrt(
        ((na - 1) * a.var(0, correction=1) + (nb - 1) * b.var(0, correction=1)) / (na + nb - 2)
    )
    return (b.mean(0) - a.mean(0)) / sp


def load_leaf_means(device: torch.device):
    """Move the leaf-mean table onto device.

    cls = red_blotch + 2 * leafroll3 -> 0 healthy, 1 RB only, 2 LR only, 3 both.

    :param device: GPU or CPU.
    :type device: torch.device
    :return: meta, wavelengths, X, glint, var, cls, row.
    :rtype: tuple
    """
    meta, w, X_np, glint_np = load_dataset()
    meta = meta.copy()
    meta["cls"] = meta.red_blotch + 2 * meta.leafroll3 # Convert to integer labels

    # Convert to tensor and move to device
    X = torch.tensor(X_np, dtype=torch.float64, device=device) 
    glint = torch.tensor(glint_np, dtype=torch.float64, device=device)
    var = torch.tensor(
        pd.Categorical(meta.variety, categories=VARIETIES).codes.copy(),
        dtype=torch.long, device=device,
    )
    cls = torch.tensor(meta.cls.to_numpy().copy(), dtype=torch.long, device=device)
    row = torch.tensor(
        pd.Categorical(meta.row).codes.copy(), dtype=torch.long, device=device,
    )
    
    return meta, w, X, glint, var, cls, row


def pca_table(Z: torch.Tensor, ev: torch.Tensor, g: torch.Tensor, brightness: torch.Tensor,
              group_key: str) -> list[dict]:
    """Build one row per PC: explained variance, eta² of g, correlation with brightness.

    :param Z: Scores, shape (n, n_pcs).
    :type Z: torch.Tensor
    :param ev: Explained-variance ratio of those PCs.
    :type ev: torch.Tensor
    :param g: Integer group labels, length n.
    :type g: torch.Tensor
    :param brightness: Raw mean reflectance per vine, length n.
    :type brightness: torch.Tensor
    :param group_key: Column name for eta² (e.g. "variety").
    :type group_key: str
    :return: One dict per PC.
    :rtype: list
    """
    rows = []
    for k in range(Z.shape[1]):
        rows.append({
            "pc": k + 1,
            "explained": float(ev[k]),
            f"eta2 {group_key}": r2(Z[:, k:k + 1], g),
            "corr with brightness": float(
                torch.corrcoef(torch.stack([Z[:, k], brightness]))[0, 1]
            ),
        })
    return rows
