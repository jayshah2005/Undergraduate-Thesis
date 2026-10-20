"""APPROACH 6 - pretrained FOUNDATION models: DOFA (hyperspectral) and DINOv2 (RGB).

Run:   python experiments/07_foundation_models.py          (full, ~30-40 min on GPU)
       python experiments/07_foundation_models.py --quick  (fold 0, 2 epochs)
Needs: `pip install timm` (already added to requirements.txt); the first run
downloads the model weights (~350 MB for DOFA, ~90 MB for DINOv2-small).

Research basis: reports/Hyperspectral grapevine transfer learning.md, option 4.
The report ranks this FOURTH on purpose: no foundation model has ever been
shown to work on leaf disease, and on a close-range benchmark the RGB-pretrained
DINOv2 + adaptor (76.6 mIoU) beat the remote-sensing hyperspectral model (60.6).
Treat this script as a fair test of that claim, not as the likely winner.

TWO FOUNDATION MODELS
  dofa    : "Dynamic One-For-All" ViT-B, pretrained with masked autoencoding on
            satellite data INCLUDING hyperspectral EnMAP scenes. Its first layer
            generates its weights from the wavelength of each band, so we can
            feed all 204 bands together with their wavelengths (in micrometres)
            and keep the pretraining intact. (Code: third_party/dofa.py, from
            TorchGeo, MIT licence.)
  dinov2  : DINOv2 ViT-S/14, pretrained on 142M ordinary photos. Like script 06
            it needs a 204 -> 3 band adaptor (1x1 conv initialised to R/G/B bands).

TWO WAYS TO USE EACH (the report: "frozen, then lightly fine-tuned")
  probe : the whole pretrained backbone is FROZEN; only a new linear output layer
          (and, for dinov2, the adaptor) is trained. Cheap and hard to overfit.
          If the pretrained features are useful, this works.
  last2 : additionally unfreeze the last 2 transformer blocks (small learning
          rate), so the model can adapt a little to leaves.

A ViT ("vision transformer") cuts the image into 16x16 (or 14x14) patches,
turns each patch into a vector ("token"), and lets all tokens exchange
information through attention layers; the average token is the image feature.
The images are upsampled from 128 to 224 pixels because these models were
pretrained at that size.

Runs produced: 07_dofa_probe, 07_dofa_last2, 07_dinov2_probe, 07_dinov2_last2.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
from dofa import DOFABase16_Weights, dofa_base_patch16_224     # noqa: E402

from common import (N_BANDS, fit_band_standardiser, fold_indices, label_mask,    # noqa: E402
                    label_matrix, load_images, load_meta, load_pixels,
                    load_wavelengths, n_folds, print_fold_summary, save_oof)
from torch_common import (LeafImageDataset, get_device, pos_weights,             # noqa: E402
                          predict_probs, seed_everything, train_model)

INPUT_SIZE = 224       # both models were pretrained on 224x224 images


class FoundationNet(nn.Module):
    def __init__(self, kind, tune, band_mean, band_std, wavelengths_nm):
        super().__init__()
        self.kind = kind
        self.register_buffer("mean", torch.tensor(band_mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(band_std).view(1, -1, 1, 1))
        self.adaptor = None

        if kind == "dofa":
            self.backbone = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
            self.wavelengths_um = [float(w) / 1000.0 for w in wavelengths_nm]   # nm -> micrometres
            feat_dim, blocks = 768, self.backbone.blocks
            late = [self.backbone.fc_norm]
        else:
            import timm
            self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                              pretrained=True, num_classes=0,
                                              img_size=INPUT_SIZE)
            feat_dim, blocks = self.backbone.num_features, self.backbone.blocks
            late = [self.backbone.norm]
            # 204 bands -> 3 pseudo-colour channels, initialised to R/G/B bands (as in 06)
            self.adaptor = nn.Conv2d(N_BANDS, 3, kernel_size=1)
            with torch.no_grad():
                self.adaptor.weight.zero_()
                self.adaptor.bias.zero_()
                for c, nm in enumerate((640.0, 550.0, 460.0)):
                    self.adaptor.weight[c, int(np.argmin(np.abs(wavelengths_nm - nm))), 0, 0] = 1.0

        self.head = nn.Linear(feat_dim, 2)       # NEW output layer, always trained

        # ---- decide what is frozen. requires_grad=False means "do not train" ----
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        if tune == "last2":
            for m in list(blocks[-2:]) + late:
                for prm in m.parameters():
                    prm.requires_grad = True
        self.tuned_backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x):                                    # x: (B, 204, 128, 128) float16
        x = x.float()
        leaf = (x.amax(dim=1, keepdim=True) > 0).float()
        z = (x - self.mean) / self.std * leaf                # per-band standardise
        if self.adaptor is not None:
            z = self.adaptor(z) * leaf                       # DINOv2 path: 204 -> 3 channels
        z = F.interpolate(z, size=INPUT_SIZE, mode="bilinear", align_corners=False)
        if self.kind == "dofa":
            feats = self.backbone.forward_features(z, self.wavelengths_um)
        else:
            feats = self.backbone(z)
        return self.head(feats)


def predict_leaves(model, images, leaf_idx, device, bs=32):
    out = []
    for s in range(0, len(leaf_idx), bs):
        x = torch.from_numpy(np.stack([np.asarray(images[i]) for i in leaf_idx[s:s + bs]]))
        out.append(predict_probs(model, x, device, amp=True))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kinds", default="dofa,dinov2")
    ap.add_argument("--tunes", default="probe,last2")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    meta, pixels = load_meta(), load_pixels(mmap=True)
    images, _ = load_images(mmap=True)
    wavelengths = load_wavelengths()
    y, mask = label_matrix(meta), label_mask(meta)
    device = get_device()
    epochs = 2 if args.quick else args.epochs
    folds = [0] if args.quick else range(n_folds())

    for kind in args.kinds.split(","):
        for tune in args.tunes.split(","):
            name = f"07_{kind}_{tune}"
            p = np.full((len(meta), 2), np.nan)
            for fold in folds:
                seed_everything(fold)
                train_idx, test_idx = fold_indices(meta, fold)
                mean, std = fit_band_standardiser(pixels, meta, train_idx)
                ds = LeafImageDataset(images, train_idx, y, mask, augment=True)
                loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=6,
                                    drop_last=True, persistent_workers=True)
                model = FoundationNet(kind, tune, mean, std, wavelengths)
                groups = [{"params": model.head.parameters(), "lr": 1e-3}]
                if model.adaptor is not None:
                    groups.append({"params": model.adaptor.parameters(), "lr": 1e-3})
                if model.tuned_backbone_params:
                    groups.append({"params": model.tuned_backbone_params, "lr": 5e-5})
                train_model(model, loader, epochs, lr=1e-3, param_groups=groups,
                            pos_weight=pos_weights(y[train_idx], mask[train_idx]),
                            device=device, amp=True, log_every=5, tag=f"{name} f{fold}")
                p[test_idx] = predict_leaves(model, images, test_idx, device)
                print_fold_summary(name, fold, meta, test_idx, p[test_idx, 0], p[test_idx, 1])
                del model
                torch.cuda.empty_cache()
            save_oof(name, meta, p[:, 0], p[:, 1], quick=args.quick)


if __name__ == "__main__":
    main()
