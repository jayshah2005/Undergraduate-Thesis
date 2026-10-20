"""APPROACH 3 - neural networks on single PIXEL spectra (MLP and 1D CNN).

Run:   python experiments/04_pixel_mlp_cnn1d.py          (full, ~10 min on GPU)
       python experiments/04_pixel_mlp_cnn1d.py --quick  (fold 0, 3 epochs)

Research basis: reports/Hyperspectral grapevine transfer learning.md,
option 2 - "Pixel-level spectral deep model from scratch with leaf-level
aggregation". In the tobacco study a small ANN on pixel spectra was the best
non-transformer model (98.9%). Spectral augmentation (random brightness and
noise) is recommended.

IDEA
  Every leaf pixel is a training example: input = its 204 reflectance values,
  output = 2 logits [red_blotch, leafroll3] using the LEAF's labels.
  Each epoch we draw PIXELS_PER_LEAF fresh random pixels from every training
  leaf. At test time we run EVERY pixel of a test leaf through the network and
  average the probabilities -> one probability per leaf.

TWO ARCHITECTURES (both small on purpose; we have only ~800 training leaves)
  mlp   : 204 -> 256 -> 64 -> 2           (fully connected)
  cnn1d : treats the spectrum as a 1-D signal and slides small filters along
          the wavelength axis, so it can pick up local SHAPES (like the red
          edge) wherever they occur.

The generic training loop lives in torch_common.py (heavily commented).
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (fit_band_standardiser, fold_indices, label_mask, label_matrix,
                    load_meta, load_pixels, N_BANDS, n_folds,
                    print_fold_summary, save_oof)
from torch_common import (PixelDataset, get_device, pos_weights, predict_pixel_leaves,
                          seed_everything, train_model)


# --------------------------------------------------------------------------
# MODELS: nn.Module classes. forward() maps a batch (B, 204) -> logits (B, 2)
# --------------------------------------------------------------------------
class PixelMLP(nn.Module):
    def __init__(self, n_bands=N_BANDS, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),            # 2 outputs: red_blotch, leafroll3 (logits)
        )

    def forward(self, x):                 # x: (B, 204)
        return self.net(x)


class PixelCNN1D(nn.Module):
    def __init__(self, n_bands=N_BANDS, dropout=0.3):
        super().__init__()
        # Conv1d expects (batch, channels, length): the spectrum is a signal of
        # length 204 with 1 channel. Each conv layer learns small filters that
        # slide along the wavelength axis. MaxPool halves the length each time.
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),                                   # 204 -> 102
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),                                   # 102 -> 51
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                                   # 51 -> 25
        )
        # We keep the position information (no global pooling): which wavelength a
        # feature occurs at matters for disease.
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(64 * 25, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):                 # x: (B, 204)
        return self.head(self.features(x.unsqueeze(1)))       # add the channel axis


ARCHS = {"mlp": PixelMLP, "cnn1d": PixelCNN1D}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--archs", default="mlp,cnn1d")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--pixels-per-leaf", type=int, default=100)
    args = ap.parse_args()

    meta, pixels = load_meta(), load_pixels(mmap=False)
    y, mask = label_matrix(meta), label_mask(meta)
    device = get_device()
    epochs = 3 if args.quick else args.epochs
    per_leaf = 20 if args.quick else args.pixels_per_leaf
    folds = [0] if args.quick else range(n_folds())
    print(f"device: {device}")

    for arch in args.archs.split(","):
        name = f"04_pixel_{arch}"
        p = np.full((len(meta), 2), np.nan)
        for fold in folds:
            seed_everything(fold)
            train_idx, test_idx = fold_indices(meta, fold)
            mean, std = fit_band_standardiser(pixels, meta, train_idx)   # TRAIN leaves only
            ds = PixelDataset(pixels, meta, train_idx, y, mask, mean, std, per_leaf, augment=True)
            loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
            model = ARCHS[arch]()
            train_model(model, loader, epochs, lr=1e-3,
                        pos_weight=pos_weights(y[train_idx], mask[train_idx]),
                        device=device, tag=f"{name} f{fold}")
            p[test_idx] = predict_pixel_leaves(model, pixels, meta, test_idx, mean, std, device)
            print_fold_summary(name, fold, meta, test_idx, p[test_idx, 0], p[test_idx, 1])
        save_oof(name, meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
