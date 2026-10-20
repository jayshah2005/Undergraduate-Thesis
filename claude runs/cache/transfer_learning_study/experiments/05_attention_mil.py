"""APPROACH 4 - attention-based Multiple-Instance Learning (MIL) on pixel bags.

Run:   python experiments/05_attention_mil.py          (full, ~5 min on GPU)
       python experiments/05_attention_mil.py --quick  (fold 0, 3 epochs)

Research basis: reports/Hyperspectral grapevine transfer learning.md,
option 2 ("optionally try attention-MIL pooling"). The report is explicit that
there is NO verified plant-disease evidence for MIL yet, so this is an
experiment, not an assumption.

THE PROBLEM IT TRIES TO SOLVE
  Script 04 labels every pixel with its leaf's label, but a diseased leaf has
  many healthy-looking pixels (symptoms are patchy). Those pixels are
  "wrong labels" that confuse the network.

  MIL fixes this by training on the whole LEAF (a "bag" of pixels) instead:
    1. encode every pixel in the bag into a small vector (shared network)
    2. an ATTENTION layer gives each pixel a weight (weights sum to 1),
       so the model can learn to focus on the pixels that show the disease
    3. the weighted average of the pixel vectors is classified
  The label is only compared with the leaf-level prediction, so healthy-looking
  pixels in a diseased leaf are simply down-weighted rather than punished.

  We use one attention head PER DISEASE (red blotch and leafroll can look
  different, so they may need to look at different pixels).

BAG SAMPLING
  A leaf has thousands of pixels; each bag is a random sample of BAG_SIZE=256
  of them (a fresh sample every time -> natural data augmentation). At test
  time we average the predictions of 8 random bags per leaf.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (fit_band_standardiser, fold_indices, label_mask, label_matrix,
                    load_meta, load_pixels, N_BANDS, n_folds, print_fold_summary,
                    save_oof)
from torch_common import (get_device, pos_weights, predict_probs, seed_everything,
                          train_model)

BAG_SIZE = 256


# --------------------------------------------------------------------------
# DATASET: one item = one bag (256 random pixels) from one training leaf
# --------------------------------------------------------------------------
class BagDataset(Dataset):
    def __init__(self, pixels, meta, leaf_idx, y, mask, mean, std, bags_per_leaf, augment):
        self.pixels, self.leaf_idx = pixels, np.asarray(leaf_idx)
        self.y, self.mask, self.mean, self.std = y, mask, mean, std
        self.bags_per_leaf, self.augment = bags_per_leaf, augment
        self.px_start, self.n_px = meta.px_start.to_numpy(), meta.n_px.to_numpy()

    def __len__(self):
        return len(self.leaf_idx) * self.bags_per_leaf

    def __getitem__(self, i):
        leaf = self.leaf_idx[i // self.bags_per_leaf]
        pick = np.sort(np.random.choice(self.n_px[leaf], BAG_SIZE, replace=False))
        x = self.pixels[self.px_start[leaf] + pick].astype(np.float32)     # (256, 204)
        if self.augment:
            x = x * np.random.uniform(0.9, 1.1)                             # brightness
        x = (x - self.mean) / self.std
        if self.augment:
            x = x + np.random.normal(0, 0.05, x.shape).astype(np.float32)   # noise
        return (torch.from_numpy(x), torch.from_numpy(self.y[leaf]),
                torch.from_numpy(self.mask[leaf]))


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, n_bands=N_BANDS, hidden=64, dropout=0.3):
        super().__init__()
        # shared pixel encoder: (B, K, 204) -> (B, K, 64). nn.Linear works on the
        # last axis, so it is applied to every pixel of every bag independently.
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, hidden), nn.ReLU())
        # attention: a small network giving each pixel 1 score PER disease
        self.attn = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 2))
        # one linear classifier per disease acting on that disease's pooled vector
        self.cls_w = nn.Parameter(torch.randn(2, hidden) * 0.05)
        self.cls_b = nn.Parameter(torch.zeros(2))
        self.drop = nn.Dropout(dropout)

    def attention(self, x):
        """Attention weights (B, K, 2); useful to SEE which pixels the model uses."""
        return torch.softmax(self.attn(self.encoder(x)), dim=1)

    def forward(self, x):                      # x: (B, K, 204)  = B bags of K pixels
        h = self.encoder(x)                    # (B, K, 64)
        a = torch.softmax(self.attn(h), dim=1)             # (B, K, 2); softmax over the K pixels
        pooled = torch.einsum("bkd,bkj->bjd", h, a)        # (B, 2, 64) attention-weighted mean
        pooled = self.drop(pooled)
        return (pooled * self.cls_w).sum(-1) + self.cls_b   # (B, 2) logits


# --------------------------------------------------------------------------
def predict_leaves(model, pixels, meta, leaf_idx, mean, std, device, n_bags=8):
    rng = np.random.default_rng(0)
    out = np.zeros((len(leaf_idx), 2))
    for k, i in enumerate(leaf_idx):
        s, n = int(meta.px_start.iloc[i]), int(meta.n_px.iloc[i])
        bags = np.stack([pixels[s + np.sort(rng.choice(n, BAG_SIZE, replace=False))]
                         for _ in range(n_bags)]).astype(np.float32)       # (8, 256, 204)
        probs = predict_probs(model, torch.from_numpy((bags - mean) / std), device)
        out[k] = probs.mean(0).numpy()          # average the 8 bags
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    meta, pixels = load_meta(), load_pixels(mmap=False)
    y, mask = label_matrix(meta), label_mask(meta)
    device = get_device()
    epochs = 3 if args.quick else args.epochs
    folds = [0] if args.quick else range(n_folds())
    name = "05_attention_mil"
    p = np.full((len(meta), 2), np.nan)

    for fold in folds:
        seed_everything(fold)
        train_idx, test_idx = fold_indices(meta, fold)
        mean, std = fit_band_standardiser(pixels, meta, train_idx)
        ds = BagDataset(pixels, meta, train_idx, y, mask, mean, std, bags_per_leaf=2, augment=True)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        model = AttentionMIL()
        train_model(model, loader, epochs, lr=1e-3,
                    pos_weight=pos_weights(y[train_idx], mask[train_idx]),
                    device=device, log_every=10, tag=f"{name} f{fold}")
        p[test_idx] = predict_leaves(model, pixels, meta, test_idx, mean, std, device)
        print_fold_summary(name, fold, meta, test_idx, p[test_idx, 0], p[test_idx, 1])
    save_oof(name, meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
