"""Building blocks of GMKL - graph-based multiple kernel learning.

Re-implementation of
  S. Hassanzadeh, H. Danyali, A. Karami, M. S. Helfroush, "A novel graph-based multiple
  kernel learning framework for hyperspectral image classification", International Journal
  of Remote Sensing 45(9):3075-3103 (2024).
No code was published; everything is read off the paper. Choices it leaves open are marked
[unspecified]; the pipeline around these pieces is 12_gmkl.py.

  1. superpixels      real-time DBSCAN superpixels (Shen et al. 2016) on 3 RGB bands, Eqs. 1-4
  2. AS kernels       adjacent-superpixel Gaussian kernels, Eq. 5
  3. EAP features     extended attribute profiles (area and standard deviation) on 3 PCs
  4. BDR graph        block diagonal representation (Lu et al. 2019), Eqs. 14-21, Algorithm 2
  5. kernel weights   graph discriminant analysis on the vectorised kernels, Eqs. 9-13

Pure functions: nothing here prints or writes files. Libraries: the dense kernel algebra is
PyTorch (GPU when given); the rest stays outside torch for the reason noted at each place -
pixel-by-pixel region growing and max-tree filtering are sequential (numba / skimage), sparse
neighbour weights use scipy.sparse, and the generalised eigenproblem uses scipy.linalg.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch
from numba import njit
from skimage.morphology import max_tree


# --------------------------------------------------------------------------
# 1. Real-time DBSCAN superpixels (paper Sec. 2.1, Eqs. 1-4)
# --------------------------------------------------------------------------
# numba, not torch: region growing visits pixels one at a time (a queue), which has no
# vectorised tensor form.
@njit(cache=True)
def _find(parent, a):
    """Union-find root of a, with path halving."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def dbscan_superpixels(rgb, mask, n_target, tc, c1=0.6, c2=0.4, c3=1.0, merge_frac=0.25):
    """Superpixel labels (H, W) int64 for the pixels in `mask`; -1 elsewhere.

    rgb: (H, W, 3) float colour image; n_target: number of superpixels wanted.
    Clustering phase (Eqs. 1, 3): pixels are visited left-to-right, top-to-bottom; an
    unlabelled pixel seeds a new superpixel that grows over 4-neighbours i of member
    pixels j while D1 = c1 * |rgb_i - rgb_j| + c2 * |rgb_i - rgb_seed| < tc, until it holds
    eps = n_pixels / n_target pixels (the paper's epsilon). tc is the colour threshold of
    Shen et al. [unspecified in GMKL].
    Merging phase (Eqs. 2, 4): superpixels are visited in the same order; one with fewer
    than merge_frac * eps pixels [threshold unspecified] absorbs the adjacent superpixel
    with the smallest D2 = |mean colour difference| + c3 * |centroid distance|.
    """
    H, W = mask.shape
    n_px = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                n_px += 1
    eps = max(1, n_px // max(1, n_target))
    labels = np.full((H, W), -1, np.int64)
    qy = np.empty(n_px, np.int64)
    qx = np.empty(n_px, np.int64)
    dys = (-1, 1, 0, 0)
    dxs = (0, 0, -1, 1)

    # Clustering phase: grow each new superpixel breadth-first from its seed (y, x). The D1 test
    # compares a candidate both with its neighbour inside the region (smooth colour changes are
    # followed) and with the seed (so the region cannot drift far from its starting colour).
    # Growth stops at eps pixels, which keeps the superpixels near the target size.
    n_lab = 0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labels[y, x] >= 0:
                continue
            lab = n_lab
            n_lab += 1
            labels[y, x] = lab
            qy[0], qx[0] = y, x
            head, tail, count, full = 0, 1, 1, False
            while head < tail and not full:
                jy, jx = qy[head], qx[head]
                head += 1
                for t in range(4):
                    iy, ix = jy + dys[t], jx + dxs[t]
                    if iy < 0 or iy >= H or ix < 0 or ix >= W:
                        continue
                    if not mask[iy, ix] or labels[iy, ix] >= 0:
                        continue
                    d_adj = 0.0
                    d_set = 0.0
                    for c in range(3):
                        d_adj += (rgb[iy, ix, c] - rgb[jy, jx, c]) ** 2
                        d_set += (rgb[iy, ix, c] - rgb[y, x, c]) ** 2
                    if c1 * np.sqrt(d_adj) + c2 * np.sqrt(d_set) < tc:
                        labels[iy, ix] = lab
                        qy[tail], qx[tail] = iy, ix
                        tail += 1
                        count += 1
                        if count >= eps:
                            full = True
                            break

    # statistics per initial superpixel and its pixel list
    size = np.zeros(n_lab)
    csum = np.zeros((n_lab, 3))
    ysum = np.zeros(n_lab)
    xsum = np.zeros(n_lab)
    for y in range(H):
        for x in range(W):
            lab = labels[y, x]
            if lab >= 0:
                size[lab] += 1
                ysum[lab] += y
                xsum[lab] += x
                for c in range(3):
                    csum[lab, c] += rgb[y, x, c]
    start = np.zeros(n_lab + 1, np.int64)
    for lab in range(n_lab):
        start[lab + 1] = start[lab] + int(size[lab])
    fill = start[:-1].copy()
    pix_y = np.empty(n_px, np.int64)
    pix_x = np.empty(n_px, np.int64)
    for y in range(H):
        for x in range(W):
            lab = labels[y, x]
            if lab >= 0:
                pix_y[fill[lab]], pix_x[fill[lab]] = y, x
                fill[lab] += 1

    # Merging phase: tiny fragments (where growth hit an edge early) are absorbed by their
    # nearest neighbour in colour + position. Union-find tracks which initial labels now form one
    # region; the nxt / last chains let us walk every pixel of a merged region to find its borders.
    parent = np.arange(n_lab)
    nxt = np.full(n_lab, -1, np.int64)                # next initial label in the same region
    last = np.arange(n_lab)                           # tail of each region's chain
    min_size = merge_frac * eps
    for lab in range(n_lab):
        if _find(parent, lab) != lab:
            continue
        while size[lab] < min_size:
            best, best_d = -1, np.inf
            m = lab
            while m != -1:                            # every pixel of the merged region
                for k in range(start[m], start[m + 1]):
                    jy, jx = pix_y[k], pix_x[k]
                    for t in range(4):
                        iy, ix = jy + dys[t], jx + dxs[t]
                        if iy < 0 or iy >= H or ix < 0 or ix >= W or labels[iy, ix] < 0:
                            continue
                        q = _find(parent, labels[iy, ix])
                        if q == lab:
                            continue
                        dc = 0.0
                        for c in range(3):
                            dc += (csum[lab, c] / size[lab] - csum[q, c] / size[q]) ** 2
                        da = ((ysum[lab] / size[lab] - ysum[q] / size[q]) ** 2
                              + (xsum[lab] / size[lab] - xsum[q] / size[q]) ** 2)
                        d2 = np.sqrt(dc) + c3 * np.sqrt(da)
                        if d2 < best_d:
                            best, best_d = q, d2
                m = nxt[m]
            if best < 0:                              # isolated fragment: nothing to merge with
                break
            parent[best] = lab                        # lab absorbs its nearest neighbour (Eq. 2)
            size[lab] += size[best]
            ysum[lab] += ysum[best]
            xsum[lab] += xsum[best]
            for c in range(3):
                csum[lab, c] += csum[best, c]
            nxt[last[lab]] = best
            last[lab] = last[best]

    # consecutive final labels
    final = np.full(n_lab, -1, np.int64)
    n_final = 0
    for lab in range(n_lab):
        r = _find(parent, lab)
        if final[r] < 0:
            final[r] = n_final
            n_final += 1
    for y in range(H):
        for x in range(W):
            if labels[y, x] >= 0:
                labels[y, x] = final[_find(parent, labels[y, x])]
    return labels


def superpixel_adjacency(labels: np.ndarray, n_sp: int) -> list[np.ndarray]:
    """For every superpixel, the sorted ids of the superpixels touching it (4-connectivity)."""
    # numpy, not torch: small per-leaf label images, part of the cached superpixel build.
    # Compare every pixel with its lower and its right neighbour; a label change is a border.
    pairs = []
    for a, b in ((labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])):
        m = (a >= 0) & (b >= 0) & (a != b)
        pairs.append(np.stack([a[m], b[m]], 1))
    p = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
    p = np.unique(np.concatenate([p, p[:, ::-1]]), axis=0)
    out = [[] for _ in range(n_sp)]
    for a, b in p:
        out[a].append(b)
    return [np.array(o, np.int64) for o in out]


# --------------------------------------------------------------------------
# 2. Gaussian kernels, including the adjacent-superpixel kernel (Eq. 5)
# --------------------------------------------------------------------------
def gaussian(X: torch.Tensor, Y: torch.Tensor, sigma2: float) -> torch.Tensor:
    """exp(-|x - y|^2 / (2 sigma^2)) for all rows of X and Y (Eq. 7, with its sign fixed)."""
    # |x|^2 + |y|^2 - 2 x.y turns the distances into one matrix product; rounding can make tiny
    # distances slightly negative, hence the clamp.
    d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return torch.exp(-d2.clamp_min(0) / (2.0 * sigma2))


def _torch_csr(m: sp.csr_matrix, device) -> torch.Tensor:
    """SciPy CSR matrix -> torch sparse CSR tensor on device."""
    m = m.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr).long(), torch.from_numpy(m.indices).long(),
                                   torch.from_numpy(m.data).float(), size=m.shape, device=device)


@torch.no_grad()
def gaussian_sandwich(A: sp.csr_matrix, B: sp.csr_matrix, means: np.ndarray, sigmas,
                      device, block: int = 2048) -> dict:
    """{sigma: A K_sigma B^T} with K_sigma = Gaussian kernel between the rows of `means`.

    This is Eq. 5: row a of A holds the normalised weights w_ik / sum_k w_ik of the
    superpixels around target a, so (A K B^T)[a, b] = sum_k sum_l w_ak w_bl K(m_k, m_l)
    / (sum w_a)(sum w_b). Only the superpixels used by A and B are touched.
    """
    # Why the sandwich: Eq. 5 is a weighted double sum over two neighbourhoods. Stacking the
    # weights of all targets into sparse A and B turns every pair at once into A K B^T, with K
    # the Gaussian kernel between superpixel means. K is built in column blocks of `block`
    # superpixels so it never has to be held in memory whole.
    ca, cb = np.unique(A.indices), np.unique(B.indices)
    Ma = torch.from_numpy(means[ca]).float().to(device)
    Mb = torch.from_numpy(means[cb]).float().to(device)
    A2, B2 = A[:, ca].tocsc(), _torch_csr(B[:, cb], device)
    out = {s: torch.zeros(A.shape[0], B.shape[0], device=device) for s in sigmas}
    nb = (Mb * Mb).sum(1)
    for s0 in range(0, len(ca), block):
        Ablk = _torch_csr(A2[:, s0:s0 + block].tocsr(), device)
        Xa = Ma[s0:s0 + block]
        d2 = ((Xa * Xa).sum(1)[:, None] + nb[None, :] - 2.0 * Xa @ Mb.T).clamp_min_(0)
        for s in sigmas:
            T = torch.sparse.mm(B2, torch.exp(-d2 / (2.0 * s * s)).T)          # (n_b, block)
            out[s] += torch.sparse.mm(Ablk, T.T.contiguous())
    return out


def neighbour_weights(centre: np.ndarray, members: list[np.ndarray], means: np.ndarray,
                      r0: float) -> sp.csr_matrix:
    """Row t: w_tk = exp(-2 r0 |centre_t - mean_k|^2) over the superpixels k in members[t],
    normalised to sum 1 (the 1 / sum_k sum_l w w factor of Eq. 5)."""
    # Why: a neighbour whose spectrum is far from the centre (e.g. across a vein or lesion edge)
    # gets a small weight, so the kernel smooths within, not across, homogeneous regions.
    # numpy + scipy.sparse, not torch: ragged neighbour lists, and scipy CSR column slicing is
    # what gaussian_sandwich needs; the float64 weights stay exactly as reported.
    rows, cols, vals = [], [], []
    for t, (c, mem) in enumerate(zip(centre, members)):
        w = np.exp(-2.0 * r0 * ((means[mem] - c) ** 2).sum(1))
        rows.append(np.full(len(mem), t))
        cols.append(mem)
        vals.append(w / w.sum())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(members), len(means)))


# --------------------------------------------------------------------------
# 3. Extended attribute profiles (paper Sec. 2.2) via max-trees
# --------------------------------------------------------------------------
# skimage max_tree + numba, not torch: a max-tree is built and walked node by node (each node's
# result depends on its parent's), which has no tensor equivalent.
@njit(cache=True)
def _node_attributes(parent, traverser, flat):
    """Area and standard deviation of the max-tree component of every node."""
    n = flat.size
    area = np.ones(n)
    s1 = flat.astype(np.float64)
    s2 = s1 * s1
    for t in range(n - 1, 0, -1):                     # leaves -> root
        p = traverser[t]
        q = parent[p]
        area[q] += area[p]
        s1[q] += s1[p]
        s2[q] += s2[p]
    std = np.sqrt(np.maximum(s2 / area - (s1 / area) ** 2, 0.0))
    return area, std


@njit(cache=True)
def _direct_filter(parent, traverser, flat, attr, thr):
    """Attribute filter, direct rule: a node is kept if attr >= thr, otherwise its pixels
    take the value of the nearest kept ancestor (= attribute thinning on a max-tree)."""
    out = np.empty_like(flat)
    root = traverser[0]
    out[root] = flat[root]
    for t in range(1, flat.size):
        p = traverser[t]
        q = parent[p]
        if flat[p] != flat[q]:                        # p is the canonical pixel of a node
            out[p] = flat[p] if attr[p] >= thr else out[q]
        else:
            out[p] = out[q]
    return out


def attribute_profile(img: np.ndarray, area_thr, std_thr) -> np.ndarray:
    """(1 + 2 * (len(area_thr) + len(std_thr)), H, W): the image, then thinnings (bright
    structures removed, max-tree) and thickenings (dark structures removed, max-tree of the
    negated image) for every area and standard-deviation threshold."""
    # A max-tree nests the bright connected components of the image by grey level; removing the
    # nodes whose area / std is below a threshold flattens small or weakly-textured bright
    # structures. Running it on -img does the same for dark structures.
    out = [img]
    for sign in (1.0, -1.0):
        f = sign * img
        parent, traverser = max_tree(f, connectivity=1)
        flat = f.ravel()
        area, std = _node_attributes(parent.ravel(), traverser, flat)
        for attr, thrs in ((area, area_thr), (std, std_thr)):
            for thr in thrs:
                out.append(sign * _direct_filter(parent.ravel(), traverser, flat, attr, thr).reshape(img.shape))
    return np.stack(out)


# --------------------------------------------------------------------------
# 4. Block diagonal representation (Eqs. 14-21, Algorithm 2)
# --------------------------------------------------------------------------
@torch.no_grad()
def block_diagonal_graph(X: np.ndarray, lam: float, gamma: float, k: int, max_iter: int = 100,
                         tol: float = 1e-4, device="cpu") -> tuple[np.ndarray, dict]:
    """Solve min 1/2|X - XZ|^2 + lam/2 |Z - W|^2 + gamma <Diag(W1) - W, S>
    s.t. diag(W) = 0, W >= 0, W = W^T, 0 <= S <= I, Tr(S) = k by alternating the three
    closed-form updates (Lu et al. 2019, BDR-B):
      S = U U^T, U = eigenvectors of L_W = Diag(W1) - W for the k smallest eigenvalues  (Eq. 19)
      Z = (X^T X + lam I)^-1 (X^T X + lam W)                                         (Eq. 20)
      W = [(A + A^T) / 2]_+ with diag 0, A = Z - gamma/lam (diag(S) 1^T - S)          (Eq. 21)
    X: (features, n) - here the M kernel values of n sample pairs. Returns W (n, n) and a log.
    """
    # Why BDR: the paper wants a graph W between samples (here: sample pairs) that has exactly k
    # connected blocks. Z is a self-expressive affinity (each column of X written as a mix of the
    # others), W is a symmetric, non-negative copy of Z, and the term <L_W, S> with S the
    # projection onto the k smallest eigenvectors of W's Laplacian is the sum of its k smallest
    # eigenvalues; driving it to zero forces k connected components (blocks). Each variable has
    # a closed-form update given the other two, so they are simply alternated.
    Xt = torch.from_numpy(X).double().to(device)
    n = Xt.shape[1]
    G = Xt.T @ Xt
    inv = torch.linalg.inv(G + lam * torch.eye(n, dtype=G.dtype, device=device))   # fixed, so inverted once
    W = torch.zeros(n, n, dtype=G.dtype, device=device)
    Z = torch.zeros_like(W)
    change = np.inf
    for it in range(1, max_iter + 1):
        L = torch.diag(W.sum(1)) - W
        _, U = torch.linalg.eigh(L)
        S = U[:, :k] @ U[:, :k].T
        Z_new = inv @ (G + lam * W)
        A = Z_new - (gamma / lam) * (torch.diag(S)[:, None] - S)
        A = A - torch.diag(torch.diag(A))                       # no self-loops: diag(W) = 0
        W_new = ((A + A.T) / 2).clamp_min(0)
        change = max((Z_new - Z).abs().max().item(), (W_new - W).abs().max().item())
        Z, W = Z_new, W_new
        if change < tol:
            break
    return W.cpu().numpy(), {"iterations": it, "final_change": change, "converged": change < tol}


# --------------------------------------------------------------------------
# 5. Kernel weights by graph discriminant analysis (Eqs. 9-13)
# --------------------------------------------------------------------------
def gda_kernel_weights(Q: np.ndarray, W: np.ndarray, pair_same: np.ndarray, penalty: str,
                       ridge: float = 1e-6) -> tuple[np.ndarray, dict]:
    """eta (M,) from P = argmin |P^T Q L Q^T P| / |P^T Q L_p Q^T P|  (Eqs. 10-12).

    Q: (M, n) vectorised kernels (one column per sample pair); W: (n, n) graph, L = D - W.
    penalty [the paper does not define its penalty graph G_p]:
      "pair" : bipartite graph joining every same-class pair to every different-class pair,
               so the denominator grows when the kernels separate the two kinds of pair
      "full" : complete graph (total scatter); the weights then use no labels at all
    eta = generalised eigenvector of the smallest non-zero eigenvalue (Eq. 12); its sign is
    chosen so that it sums to a positive number, negatives are clipped to 0 and it is
    normalised to sum 1 (Eq. 9 needs eta >= 0, sum 1) [how is unspecified].
    """
    # Why: L (from the BDR graph) measures how much the combined kernel values differ between
    # pairs the graph puts together; L_p (penalty graph) measures how much they differ between
    # pairs that should look different. The eta minimising the ratio makes pairs in one BDR
    # block look alike while keeping the penalty pairs apart.
    # numpy / scipy, not torch: torch has no generalised symmetric eigensolver (eigh(A, B)),
    # and the small (n_pairs x n_pairs) products stay in the same BLAS as the reported run.
    L = np.diag(W.sum(1)) - W
    if penalty == "pair":
        s, d = pair_same.astype(float), (~pair_same).astype(float)
        Wp = np.outer(s, d) + np.outer(d, s)
    elif penalty == "full":
        Wp = np.ones((len(pair_same), len(pair_same))) - np.eye(len(pair_same))
    else:
        raise ValueError(penalty)
    Lp = np.diag(Wp.sum(1)) - Wp
    Sw, Sp = Q @ L @ Q.T, Q @ Lp @ Q.T
    Sp = Sp + ridge * np.trace(Sp) / len(Sp) * np.eye(len(Sp))   # tiny ridge: eigh(Sw, Sp) needs Sp positive definite
    evals, evecs = scipy.linalg.eigh(Sw, Sp)
    # Zero eigenvalues belong to directions with no within-graph variation at all (e.g. kernels
    # that are constant over the pairs); they are uninformative, so take the smallest non-zero one.
    nonzero = np.where(evals > 1e-10 * max(abs(evals).max(), 1e-300))[0]
    j = nonzero[0] if len(nonzero) else 0
    # An eigenvector's sign is arbitrary; flip it so the weights mostly point the positive way,
    # then clip negatives because a kernel mix with negative weights need not be a valid kernel.
    # If nothing positive is left, fall back to equal weights (reported as uniform_fallback).
    p = evecs[:, j] * (1.0 if evecs[:, j].sum() >= 0 else -1.0)
    eta = np.clip(p, 0, None)
    fallback = eta.sum() <= 0
    eta = np.full(len(p), 1.0 / len(p)) if fallback else eta / eta.sum()
    return eta, {"eigenvalue": float(evals[j]), "raw_vector": p.tolist(),
                 "n_clipped": int((p < 0).sum()), "uniform_fallback": bool(fallback)}
