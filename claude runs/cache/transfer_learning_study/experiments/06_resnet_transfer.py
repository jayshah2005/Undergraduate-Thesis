"""APPROACH 5 - TRANSFER LEARNING: ImageNet ResNet-18 on whole leaf images,
pretrained versus trained from scratch.

Run:   python experiments/06_resnet_transfer.py          (full, ~15-25 min on GPU)
       python experiments/06_resnet_transfer.py --quick  (fold 0, 2 epochs)

Research basis: reports/Hyperspectral grapevine transfer learning.md, option 3.
This is the direct answer to "is transfer learning worth it?": the SAME network,
the SAME folds, once starting from ImageNet weights and once from random
weights. Whatever difference you see is due to pretraining alone.
(Perez & Maji 2021: with ~1000 samples a pretrained net + learned linear channel
adaptor beat the from-scratch one by ~13 points on EuroSAT - but that was
satellite images, not leaves; this script tests whether it holds here.)

THE PROBLEM
  ResNet-18 was trained on ordinary colour photos: 3 input channels (R,G,B).
  Our leaf images have 204 channels (bands). We cannot feed 204 channels to the
  pretrained first layer, so we put a tiny trainable layer in front:

        204-band image --[1x1 convolution, 204 -> 3]--> 3-channel image --> ResNet-18

  A 1x1 convolution is just "each output channel = a weighted sum of the 204
  bands at that pixel" - a learnable way to make a false-colour image.
  It is INITIALISED to pick the bands closest to red (640 nm), green (550 nm) and
  blue (460 nm), so training starts from an ordinary-looking RGB picture and the
  pretrained filters make sense from step 1. Training then adjusts the weights
  so the 3 channels can carry the disease-relevant bands. This 'adaptor'
  gets a 10x larger learning rate than the rest of the network (the report
  recommends this).

INPUT
  Leaves are cut-outs on a black (zero) background, shrunk to fit 128x128 if
  larger, otherwise left at real size. The mask of leaf pixels is re-derived
  inside the model (background is exactly 0) so the black background stays black.

AUGMENTATION (done on the fly in the Dataset): random horizontal/vertical flip,
random 90-degree rotation, random +-10% brightness. At test time we average the
prediction over 4 flips/rotations ("test-time augmentation").

OPTIONS
  --modes pretrained,scratch      which runs to do (default both)
  --adaptor learned|fixed_rgb     'fixed_rgb' freezes the adaptor at the plain
                                  R/G/B band picks = a cheap pseudo-RGB control
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from common import (IMG_SIZE, N_BANDS, fit_band_standardiser, fold_indices,
                    label_mask, label_matrix, load_images, load_meta, load_pixels,
                    load_wavelengths, n_folds, print_fold_summary, save_oof)
from torch_common import (LeafImageDataset, get_device, pos_weights, predict_probs, seed_everything,
                          train_model)


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
class LeafResNet(nn.Module):
    def __init__(self, band_mean, band_std, wavelengths, pretrained, adaptor="learned"):
        super().__init__()
        # Buffers = fixed tensors that move to the GPU with the model but are not trained.
        self.register_buffer("mean", torch.tensor(band_mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(band_std).view(1, -1, 1, 1))

        # --- the adaptor: 204 bands -> 3 channels, initialised as pseudo-RGB ---
        self.reducer = nn.Conv2d(N_BANDS, 3, kernel_size=1)
        with torch.no_grad():
            self.reducer.weight.zero_()
            self.reducer.bias.zero_()
            for c, nm in enumerate((640.0, 550.0, 460.0)):          # R, G, B
                self.reducer.weight[c, int(np.argmin(np.abs(wavelengths - nm))), 0, 0] = 1.0
        if adaptor == "fixed_rgb":
            for prm in self.reducer.parameters():
                prm.requires_grad = False

        # --- the backbone: ResNet-18, optionally with ImageNet weights ---
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        # replace the 1000-class ImageNet output layer with our 2 outputs
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):                                  # x: (B, 204, 128, 128) float16
        x = x.float()
        leaf = (x.amax(dim=1, keepdim=True) > 0).float()    # (B,1,H,W): 1 where there is leaf
        z = (x - self.mean) / self.std * leaf               # per-band standardise, keep background 0
        z = self.reducer(z) * leaf                          # 204 -> 3 channels
        return self.backbone(z)                             # (B, 2) logits


def predict_leaves(model, images, leaf_idx, device, bs=32):
    """Test-time augmentation: average over 4 views (identity, h-flip, v-flip, rot180)."""
    out = []
    for s in range(0, len(leaf_idx), bs):
        x = torch.from_numpy(np.stack([np.asarray(images[i]) for i in leaf_idx[s:s + bs]]))
        views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.rot90(x, 2, (2, 3))]
        out.append(torch.stack([predict_probs(model, v, device, amp=True) for v in views]).mean(0))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--modes", default="pretrained,scratch")
    ap.add_argument("--adaptor", default="learned", choices=["learned", "fixed_rgb"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4, help="backbone learning rate")
    args = ap.parse_args()

    meta, pixels = load_meta(), load_pixels(mmap=True)
    images, _ = load_images(mmap=True)
    wavelengths = load_wavelengths()
    y, mask = label_matrix(meta), label_mask(meta)
    device = get_device()
    epochs = 2 if args.quick else args.epochs
    folds = [0] if args.quick else range(n_folds())
    print(f"device: {device}")

    for mode in args.modes.split(","):
        name = f"06_resnet18_{mode}" + ("" if args.adaptor == "learned" else "_fixedrgb")
        p = np.full((len(meta), 2), np.nan)
        for fold in folds:
            seed_everything(fold)
            train_idx, test_idx = fold_indices(meta, fold)
            mean, std = fit_band_standardiser(pixels, meta, train_idx)      # TRAIN leaves only
            ds = LeafImageDataset(images, train_idx, y, mask, augment=True)
            loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=6,
                                drop_last=True, persistent_workers=True)
            model = LeafResNet(mean, std, wavelengths, pretrained=(mode == "pretrained"),
                               adaptor=args.adaptor)
            groups = [
                {"params": model.backbone.parameters(), "lr": args.lr},
                {"params": model.reducer.parameters(), "lr": args.lr * 10},   # 10x for the adaptor
            ]
            train_model(model, loader, epochs, lr=args.lr, param_groups=groups,
                        pos_weight=pos_weights(y[train_idx], mask[train_idx]),
                        device=device, amp=True, log_every=5, tag=f"{name} f{fold}")
            p[test_idx] = predict_leaves(model, images, test_idx, device)
            print_fold_summary(name, fold, meta, test_idx, p[test_idx, 0], p[test_idx, 1])
        save_oof(name, meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
