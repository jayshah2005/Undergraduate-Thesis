"""APPROACH 7 - SELF-SUPERVISED pretraining on our own spectra, then fine-tune.

Run:   python experiments/08_self_supervised_pretraining.py          (full, ~10 min)
       python experiments/08_self_supervised_pretraining.py --quick  (fold 0, tiny)

Research basis: reports/Hyperspectral grapevine transfer learning.md, option 5.
The report ranks this LAST and says to expect "a null result as often as a gain":
masked-autoencoder gains in the literature come from ~22,000 satellite tiles, and
nobody has shown the effect for ~1000 leaves. This script tests it fairly.

THE IDEA (transfer learning without any outside data)
  Step 1 - PRETRAIN WITHOUT LABELS. Take pixel spectra, hide random blocks of
           bands (about half of them), and train a network to RECONSTRUCT the
           hidden bands from the visible ones. To do that well the encoder must
           learn how the bands relate to each other (e.g. how chlorophyll
           absorption in the red predicts the red edge). This is a "masked
           autoencoder". No disease labels are used, so we can learn from
           ALL pixels of the training leaves (millions of examples).
  Step 2 - FINE-TUNE WITH LABELS. Throw away the decoder, attach a small
           classification head to the pretrained encoder, and train on the
           disease labels exactly like script 04.
  CONTROL - the SAME architecture trained from RANDOM weights (--modes
           scratch). The difference between 'ssl' and 'scratch' is the effect of
           pretraining and nothing else.

NO LEAKAGE: the pretraining in each fold sees only the pixels of that fold's
TRAINING leaves (no labels, but also no test pixels).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (N_BANDS, fit_band_standardiser, fold_indices, label_mask,
                    label_matrix, load_meta, load_pixels, n_folds,
                    print_fold_summary, sample_pixels_per_leaf, save_oof)
from torch_common import (PixelDataset, get_device, pos_weights, predict_pixel_leaves,
                          seed_everything, train_model)

CHUNK = 6          # hide bands in contiguous blocks of 6 (~18 nm), not one by one;
                   # neighbouring bands are nearly identical so single hidden bands are too easy
MASK_RATIO = 0.5
RUN_NAMES = {"ssl": "08_ssl_pretrained", "scratch": "08_ssl_scratch"}


class SpectralEncoder(nn.Module):
    """204 bands -> 128-number summary of the pixel."""

    def __init__(self, n_bands=N_BANDS, latent=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_bands, 256), nn.LayerNorm(256), nn.GELU(),
                                 nn.Linear(256, latent), nn.LayerNorm(latent), nn.GELU())

    def forward(self, x):
        return self.net(x)


class MaskedAutoencoder(nn.Module):
    """Encoder + decoder. The decoder is used ONLY during pretraining."""

    def __init__(self):
        super().__init__()
        self.encoder = SpectralEncoder()
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, N_BANDS))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PixelClassifier(nn.Module):
    """Encoder (pretrained or not) + small head -> 2 logits [red_blotch, leafroll3]."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.encoder(x))


def random_band_mask(batch, device):
    """(batch, 204) float mask: 1 = band hidden. Whole chunks of CHUNK bands at a time."""
    n_chunks = int(np.ceil(N_BANDS / CHUNK))
    chunk_mask = (torch.rand(batch, n_chunks, device=device) < MASK_RATIO).float()
    return chunk_mask.repeat_interleave(CHUNK, dim=1)[:, :N_BANDS]


def pretrain(mae, X, epochs, device, tag, batch_size=1024, lr=1e-3):
    """Train the masked autoencoder on standardised pixels X (n, 204), NO labels.

    Because X is small enough to sit on the GPU, we skip the DataLoader and just
    shuffle indices and slice mini-batches by hand - the loop is otherwise the
    same one as in torch_common.train_model.
    """
    mae.to(device).train()
    X = torch.from_numpy(X).to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * (len(X) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(X), device=device)           # new random order each epoch
        total = 0.0
        for s in range(0, len(X) - batch_size + 1, batch_size):
            x = X[perm[s:s + batch_size]]
            hidden = random_band_mask(len(x), device)           # 1 where the band is hidden
            recon = mae(x * (1 - hidden))                       # network sees hidden bands as 0
            # loss = squared error on the HIDDEN bands only (it can't cheat by copying visible ones)
            loss = (((recon - x) ** 2) * hidden).sum() / hidden.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"   [{tag}] pretrain epoch {epoch + 1:3d}/{epochs}  "
                  f"masked-band MSE {total / (len(X) // batch_size):.4f}  ({time.time() - t0:.0f}s)")
    return mae.encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--modes", default="ssl,scratch")
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--pretrain-pixels-per-leaf", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=15, help="fine-tuning epochs")
    args = ap.parse_args()

    meta, pixels = load_meta(), load_pixels(mmap=False)
    y, mask = label_matrix(meta), label_mask(meta)
    device = get_device()
    folds = [0] if args.quick else range(n_folds())
    pre_epochs = 2 if args.quick else args.pretrain_epochs
    pre_px = 50 if args.quick else args.pretrain_pixels_per_leaf
    ft_epochs = 2 if args.quick else args.epochs
    preds = {m: np.full((len(meta), 2), np.nan) for m in args.modes.split(",")}

    for fold in folds:
        train_idx, test_idx = fold_indices(meta, fold)
        mean, std = fit_band_standardiser(pixels, meta, train_idx)
        X_pre = None
        for mode, p in preds.items():
            name = RUN_NAMES[mode]
            seed_everything(fold)
            if mode == "ssl":
                if X_pre is None:        # pixels of TRAIN leaves only, labels never touched
                    X_pre, _ = sample_pixels_per_leaf(pixels, meta, train_idx, pre_px, seed=fold)
                    X_pre = ((X_pre - mean) / std).astype(np.float32)
                mae = MaskedAutoencoder()
                encoder = pretrain(mae, X_pre, pre_epochs, device, tag=f"{name} f{fold}")
            else:
                encoder = SpectralEncoder()            # same architecture, random weights
            model = PixelClassifier(encoder)

            ds = PixelDataset(pixels, meta, train_idx, y, mask, mean, std,
                              per_leaf=20 if args.quick else 100, augment=True)
            loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
            groups = [{"params": model.encoder.parameters(), "lr": 3e-4},   # gentle: keep what was learned
                      {"params": model.head.parameters(), "lr": 1e-3}]
            train_model(model, loader, ft_epochs, lr=1e-3, param_groups=groups,
                        pos_weight=pos_weights(y[train_idx], mask[train_idx]),
                        device=device, tag=f"{name} f{fold}")
            p[test_idx] = predict_pixel_leaves(model, pixels, meta, test_idx, mean, std, device)
            print_fold_summary(name, fold, meta, test_idx, p[test_idx, 0], p[test_idx, 1])

    for mode, p in preds.items():
        save_oof(RUN_NAMES[mode], meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
