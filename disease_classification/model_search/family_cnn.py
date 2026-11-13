"""CNN candidates with a clean validation row, saved in the harness prediction format.

Per fold k (test row k, validation row k+1, training rows k+2, k+3, k+4):
1. train on rows k+3, k+4 and early-stop on row k+2 (lowest weighted loss, patience 15) -> best epoch e;
2. retrain from the same initial weights on all 3 training rows for e epochs
   (Goodfellow, Bengio & Courville 2016, Deep Learning, Algorithm 7.2);
3. predict the validation and test rows, which neither step saw.
Everything else is the repo recipe (disease_classification/cnn/train.py): pixel SNV, rotations / flips,
AdamW lr 1e-3 / wd 1e-2, batch 8, dropout 0.5, up to 80 epochs. Loss: binary cross-entropy with
per-leaf weights that balance class (within one variety) or class within each variety (pooled: group
balancing, Idrissi et al. 2022), so the variety predicts nothing.

usage: family_cnn.py <backbone> <pooling> <bands: all|rgb|vis> <mode: within|pooled> [seeds] [lr wd dropout]
(lr, wd, dropout given: the run is named "tuned:cnn:..." with them in the name.)
backbone "film_small_cnn": the small CNN conditioned on variety with FiLM (film.py); pooled mode only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]
for p in (R, R / "disease_classification/logistic_regression", R / "disease_classification/cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.cnn_common import augment  # noqa: E402
from logistic_regression import fold_of, load_meta, task_labels, usable_leaves  # noqa: E402
from model import LeafCNN  # noqa: E402
from run_experiments import load  # noqa: E402
from train import preprocess  # noqa: E402
from harness import PRED, TASKS, VARIETIES, group_weights  # noqa: E402

BATCH, LR, WD, DROPOUT, MAX_EPOCHS, PATIENCE = 8, 1e-3, 1e-2, 0.5, 80, 15


def band_subset(bands: str, w: np.ndarray) -> np.ndarray:
    """Indices of the bands a candidate sees."""
    if bands == "all":
        return np.arange(len(w))
    if bands == "rgb":
        return np.array([int(np.abs(w - nm).argmin()) for nm in (640.0, 550.0, 460.0)])
    if bands == "vis":   # Sawyer et al. 2023 imaged 510-710 nm
        return np.flatnonzero((w >= 510) & (w <= 710))
    raise ValueError(bands)


def run(model, x, m, vt, j):
    """Logits of leaves j; FiLM models also get each leaf's variety index."""
    return model(x, m, vt[j])[0] if vt is not None else model(x, m)[0]


def predict(model, X, mask, idx, vt=None, batch=16):
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            lo = run(model, preprocess(X[j], mask[j]), mask[j], vt, j)
            out.append(lo)
    return torch.cat(out)


def train(X, mask, y, w, tr, n_epochs, seed, backbone, pooling, stop=None, vt=None, hp=(LR, WD, DROPOUT)):
    """Train from seed's initial weights; with stop=(idx, weights) early-stop on it, else run n_epochs.

    :return: (model, best epoch)
    """
    dev = X.device
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    if backbone == "film_small_cnn":
        from film import FiLMLeafCNN
        model = FiLMLeafCNN(X.shape[1], pooling, hp[2]).to(dev)
    else:
        model = LeafCNN(X.shape[1], pooling, hp[2], backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp[0], weight_decay=hp[1])
    best = (float("inf"), 0, None)
    for epoch in range(1, (MAX_EPOCHS if stop is not None else n_epochs) + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g, device=dev)]
        for s in range(0, len(order), BATCH):
            j = order[s:s + BATCH]
            x, m = augment(preprocess(X[j], mask[j]), mask[j], g)
            logit = run(model, x, m, vt, j)
            loss = (F.binary_cross_entropy_with_logits(logit, y[j], reduction="none") * w[j]).sum() / w[j].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if stop is not None:
            si, sw = stop
            lv = predict(model, X, mask, si, vt)
            vloss = ((F.binary_cross_entropy_with_logits(lv, y[si], reduction="none") * sw).sum() / sw.sum()).item()
            if vloss < best[0]:
                best = (vloss, epoch, None)
            elif epoch - best[1] >= PATIENCE:
                break
    return model, (best[1] if stop is not None else n_epochs)


def main(backbone: str, pooling: str, bands: str, mode: str, n_seeds: int = 3, hp: tuple | None = None) -> None:
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    w_nm = np.load(R / "cache/disease_search/features.npz")["wavelengths"]
    name = f"cnn:{backbone}_{pooling}_{bands}" + ("" if n_seeds == 3 else f"_s{n_seeds}") + f"@{mode}"
    if hp is not None:
        name = f"tuned:cnn:{backbone}_{pooling}_{bands}_lr{hp[0]:g}_wd{hp[1]:g}_do{hp[2]:g}@{mode}"
    hp = hp or (LR, WD, DROPOUT)
    units = [[v] for v in VARIETIES] if mode == "within" else [list(VARIETIES)]
    arrs, t0 = {}, time.time()
    for vs in units:
        pool = vs[0] if len(vs) == 1 else "baco_chardonnay_vidal"
        d = load(pool, "all", dev)
        # Keep the 204-band canvas as is (no copy: the pooled one is 13.7 GB); subsets are copies.
        X = d["X"] if bands == "all" else d["X"][:, torch.as_tensor(band_subset(bands, w_nm), device=dev)].contiguous()
        del d["X"]
        torch.cuda.empty_cache()
        meta, mask = d["meta"], d["mask"]
        assert list(meta.path) == list(load_meta(pool).path)
        fold, var = fold_of(meta), meta.variety.to_numpy()
        vt = (torch.tensor([VARIETIES.index(v) for v in var], device=dev)
              if backbone == "film_small_cnn" else None)
        assert vt is None or mode == "pooled"
        for task in TASKS:
            yn = task_labels(meta, task)
            keep = usable_leaves(meta, yn)
            y = torch.tensor(yn, dtype=torch.float32, device=dev)
            val_p = np.full((n_seeds, len(yn)), np.nan)
            test_p = np.full((n_seeds, len(yn)), np.nan)
            for k in range(5):
                idx = np.flatnonzero(keep)
                rows = fold[idx]
                va, te = idx[rows == (k + 1) % 5], idx[rows == k]
                tr = idx[(rows != k) & (rows != (k + 1) % 5)]
                st = tr[fold[tr] == (k + 2) % 5]
                tr2 = tr[fold[tr] != (k + 2) % 5]
                assert len(set(va) | set(te) | set(tr)) == len(va) + len(te) + len(tr)
                w_all = torch.tensor(group_weights(yn[tr], var[tr]), dtype=torch.float32, device=dev)
                w2 = torch.tensor(group_weights(yn[tr2], var[tr2]), dtype=torch.float32, device=dev)
                ws = torch.tensor(group_weights(yn[st], var[st]), dtype=torch.float32, device=dev)
                wfull = torch.zeros(len(yn), device=dev)
                for s in range(n_seeds):
                    seed = 100 * s + k
                    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
                    wfull.zero_()
                    wfull[t(tr2)] = w2
                    _, e = train(X, mask, y, wfull, t(tr2), None, seed, backbone, pooling, stop=(t(st), ws), vt=vt, hp=hp)
                    wfull.zero_()
                    wfull[t(tr)] = w_all
                    model, _ = train(X, mask, y, wfull, t(tr), e, seed, backbone, pooling, vt=vt, hp=hp)
                    val_p[s, va] = torch.sigmoid(predict(model, X, mask, t(va), vt)).cpu().numpy()
                    test_p[s, te] = torch.sigmoid(predict(model, X, mask, t(te), vt)).cpu().numpy()
            assert not np.isnan(val_p[:, keep]).any() and not np.isnan(test_p[:, keep]).any()
            vp, tp = val_p.mean(0), test_p.mean(0)
            for v in vs:
                m = keep & (var == v)
                arrs.update({f"{task}|{v}|val": vp[m], f"{task}|{v}|test": tp[m], f"{task}|{v}|y": yn[m],
                             f"{task}|{v}|row": fold[m]})
        del X, mask, d
        torch.cuda.empty_cache()
    np.savez(PRED / f"{name}.npz", **arrs)
    print(f"[saved] {name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 3, tuple(map(float, a[5:8])) if len(a) > 7 else None)
