"""Frozen DINOv2 embeddings of each leaf rendered as a 3-band image (Oquab et al., DINOv2).

Renders (bands as R, G, B):
- rgb: 640 / 550 / 460 nm (true colour, the repo's RGB picks)
- cir: 800 / 670 / 550 nm (colour infrared)

Each band is divided by its 99th percentile over all leaf pixels of all leaves (one global scale, so
brightness differences between leaves are kept), clipped to [0, 1], centred on a 224 x 224 black canvas
and normalised with the ImageNet mean / SD. Features are averaged over the 8 rotations and flips:
the CLS token and the mean of the patch tokens whose 14 x 14 patch is mostly leaf.
Saved to cache/disease_search/dino_<model>_<render>.npz in load_meta("all") order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "disease_classification/logistic_regression"):
    sys.path.insert(0, str(p))
from helpers.data import load_cubes, load_dataset  # noqa: E402
from logistic_regression import load_meta  # noqa: E402

OUT = ROOT / "cache/disease_search"
RENDERS = {"rgb": (640.0, 550.0, 460.0), "cir": (800.0, 670.0, 550.0)}
SIDE, PATCH = 224, 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dihedral(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 rotations / flips of (B, C, H, W)."""
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (2, 3))
        out += [r, r.flip(3)]
    return out


@torch.no_grad()
def main(model_name: str = "vit_small_patch14_dinov2.lvd142m") -> None:
    import timm
    dev = torch.device("cuda")
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=SIDE).to(dev).eval()
    meta = load_meta("all")
    w = load_dataset()[1]
    bands = {r: [int(np.abs(w - nm).argmin()) for nm in nms] for r, nms in RENDERS.items()}
    all_b = sorted({b for bs in bands.values() for b in bs})

    # Pass 1: keep only the needed bands of every leaf (small), and the mask.
    imgs, masks, paths = [], [], []
    for v in meta.variety.unique():
        m, cubes = load_cubes(v)
        paths += list(m.path)
        for c in cubes:
            msk = (c != 0).any(0)
            imgs.append({b: c[b] for b in all_b})
            masks.append(msk)
        del cubes
    assert paths == list(meta.path)
    # One global scale per band: its 99th percentile over every leaf pixel.
    scale = {b: np.percentile(np.concatenate([im[b][mk] for im, mk in zip(imgs, masks)]), 99) for b in all_b}

    for render, bs in bands.items():
        cls_f, mean_f = [], []
        for im, mk in zip(imgs, masks):
            h, wd = mk.shape
            x = np.zeros((3, SIDE, SIDE), np.float32)
            m = np.zeros((SIDE, SIDE), np.float32)
            top, left = (SIDE - h) // 2, (SIDE - wd) // 2
            for ch, b in enumerate(bs):
                x[ch, top:top + h, left:left + wd] = np.clip(im[b] / scale[b], 0, 1) * mk
            m[top:top + h, left:left + wd] = mk
            xt = (torch.from_numpy(x)[None] - MEAN) / STD
            xt = xt * torch.from_numpy(m)[None, None] + (-MEAN / STD) * (1 - torch.from_numpy(m)[None, None])
            mt = torch.from_numpy(m)[None, None]
            cs, ms = [], []
            for xa, ma in zip(dihedral(xt.to(dev)), dihedral(mt.to(dev))):
                tok = model.forward_features(xa)                       # (1, 1 + n_reg + P, D)
                patches = tok[:, model.num_prefix_tokens:]             # (1, P, D)
                cover = F.avg_pool2d(ma, PATCH).flatten(1)             # (1, P): leaf share of each patch
                leaf = cover > 0.5 if (cover > 0.5).any() else cover > 0
                cs.append(tok[0, 0])
                ms.append(patches[0][leaf[0]].mean(0))
            cls_f.append(torch.stack(cs).mean(0).cpu().numpy())
            mean_f.append(torch.stack(ms).mean(0).cpu().numpy())
        tag = model_name.split(".")[0].replace("vit_", "").replace("_patch14_dinov2", "")
        assert np.isfinite(np.stack(mean_f)).all() and np.isfinite(np.stack(cls_f)).all()
        np.savez(OUT / f"dino_{tag}_{render}.npz", cls=np.stack(cls_f), mean=np.stack(mean_f), paths=np.array(paths))
        print(f"[saved] dino_{tag}_{render}.npz", np.stack(cls_f).shape, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
