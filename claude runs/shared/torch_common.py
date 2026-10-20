"""PyTorch helpers: device / seeding for every pipeline, plus the training loop,
datasets and pixel-level prediction used by the archived deep-learning scripts
(04-08 in ../cache/transfer_learning_study/experiments/).

If you are new to PyTorch, READ THIS FILE FIRST. It contains the one training
loop that every archived neural-network script reuses, with every step explained.

Logging: nothing here prints except train_model's progress lines, which go to
the `log` argument (a logging.Logger) when given, else are printed as before.
NumPy random calls inside the datasets are kept on purpose (see the comments):
the archived results were produced with those random streams.

PYTORCH IN 60 SECONDS
---------------------
* Tensor      : like a numpy array, but it can live on the GPU and remembers
                how it was computed so gradients can be calculated.
* nn.Module   : a model. You define layers in __init__ and describe how data
                flows through them in forward(x). Calling model(x) runs forward.
* Dataset     : an object with __len__ (how many examples) and __getitem__(i)
                (return example i). Ours return (x, y, mask).
* DataLoader  : takes a Dataset and serves it in shuffled mini-batches.
* Logits      : the raw number a model outputs before squashing. sigmoid(logit)
                gives a probability between 0 and 1. We train on logits
                (numerically safer) and apply sigmoid only when we want a probability.
* Loss        : one number saying how wrong the predictions are. Training =
                repeatedly nudging the weights to make the loss smaller.
* Optimizer   : the thing that nudges the weights (AdamW here).

THE TWO OUTPUTS
---------------
Every model outputs 2 logits per example: [red_blotch, leafroll3]. They are
two independent yes/no questions (a leaf can be both), so we use a binary
cross-entropy loss on each output separately ("multi-label" classification).
"healthy" is then simply "both probabilities are low".
"""
from __future__ import annotations

import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import leaf_pixels


def get_device() -> torch.device:
    """Use the GPU if there is one (much faster), otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Make runs repeatable: fix every random number generator we use (Python, NumPy - the
    datasets below draw from it - and torch on CPU and every GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pos_weights(y: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """How much to up-weight the rare class, per label: (#negatives / #positives).

    Without this, a model that always says "negative" looks good on a
    label that is mostly negative. Computed from TRAIN leaves only.
    y, mask: (n_leaves, 2) labels and label mask (common.label_matrix / label_mask);
    returns a (2,) float32 tensor for BCEWithLogits' pos_weight. max(.., 1) avoids 0 / 0.
    """
    # why not torch: two counts on a tiny NumPy array; the result is converted to a tensor below
    w = []
    for j in range(2):
        yj = y[mask[:, j], j]
        w.append(max((yj == 0).sum(), 1) / max((yj == 1).sum(), 1))
    return torch.tensor(w, dtype=torch.float32)


def masked_bce(logits, y, mask, pos_weight):
    """Binary cross-entropy for 2 labels, ignoring entries where mask is False.

    logits, y, mask : (batch, 2) tensors.
    mask is False for (Cab Franc leaf, leafroll label) because that label has
    only 1 positive example in the whole variety.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight.to(logits.device), reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def train_model(model: nn.Module, loader: DataLoader, epochs: int, lr: float,
                pos_weight: torch.Tensor, device: torch.device,
                param_groups=None, weight_decay: float = 1e-2,
                amp: bool = False, log_every: int = 5, tag: str = "",
                log: logging.Logger | None = None) -> nn.Module:
    """The training loop. Batches of (x, y, mask) come out of `loader`.

    param_groups: optional list of dicts like [{"params": ..., "lr": ...}, ...]
                  to give different parts of the model different learning rates.
    amp         : "automatic mixed precision" - runs the maths in bfloat16 on
                  the GPU, which is faster and uses less memory.
    log_every, tag, log : one progress line (mean train loss) after the first
                  epoch, every `log_every` epochs and the last; it goes to `log`
                  if given, else is printed. Returns the trained model (same object).
    NOTE: there is deliberately NO early stopping or "pick the best epoch" on
    the test fold; that would leak test information. We train for a fixed
    number of epochs and use the final model.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups if param_groups else model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    # The learning rate follows a cosine curve from `lr` down to ~0 over training.
    # (One scheduler step per batch.)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()                       # switch on dropout / batch-norm training behaviour
        running, n = 0.0, 0
        for x, y, mask in loader:
            # 1. move this batch from CPU memory to the GPU
            x, y, mask = x.to(device, non_blocking=True), y.to(device), mask.to(device)

            # 2. forget the gradients from the previous batch
            optimizer.zero_grad(set_to_none=True)

            # 3. forward pass: the model's predictions (logits) for this batch
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                logits = model(x)
            loss = masked_bce(logits.float(), y, mask, pos_weight)

            # 4. backward pass: work out how each weight should change to reduce the loss
            loss.backward()

            # 5. clip huge gradients (stability), then 6. update the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            running += loss.item() * len(y)
            n += len(y)
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch + 1 == epochs:
            msg = (f"   [{tag}] epoch {epoch + 1:3d}/{epochs}  "
                   f"train loss {running / n:.4f}  ({time.time() - t0:.0f}s)")
            (log.info if log is not None else print)(msg)
    return model


@torch.no_grad()      # no gradients needed when only predicting: saves memory and time
def predict_probs(model: nn.Module, x: torch.Tensor, device, amp: bool = False) -> torch.Tensor:
    """Run the model on a batch x and return probabilities (batch, 2) on the CPU (float32)."""
    model.eval()                            # switch OFF dropout, use stored batch-norm statistics
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp and device.type == "cuda"):
        logits = model(x.to(device, non_blocking=True))
    return torch.sigmoid(logits.float()).cpu()      # sigmoid: logit -> probability


class LeafLabels:
    """Tiny helper that turns leaf row-positions into (y, mask) float tensors."""

    def __init__(self, y: np.ndarray, mask: np.ndarray):
        self.y = torch.from_numpy(y.astype(np.float32))
        self.mask = torch.from_numpy(mask)


# --------------------------------------------------------------------------
# DATASET: one item = one whole leaf image (used by scripts 06, 07)
# --------------------------------------------------------------------------
class LeafImageDataset(Dataset):
    """Item i = (image (204, 128, 128) float16, y (2,), mask (2,)) of leaf leaf_idx[i].

    images: common.load_images()[0] (memmap is fine); y, mask: (n_leaves, 2) arrays for ALL
    leaves, indexed by leaf position. augment=True adds random flips, 90-degree rotations and
    a +-10% brightness change (leaf orientation and illumination carry no disease information).
    """

    def __init__(self, images, leaf_idx, y, mask, augment):
        self.images, self.leaf_idx = images, np.asarray(leaf_idx)
        self.y, self.mask, self.augment = y, mask, augment

    def __len__(self):
        return len(self.leaf_idx)

    def __getitem__(self, i):
        leaf = self.leaf_idx[i]
        x = torch.from_numpy(np.array(self.images[leaf]))          # (204,128,128) float16 (copy: memmap is read-only)
        if self.augment:
            # why not torch: np.random keeps the archived runs' random stream (seeded by seed_everything)
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])                         # horizontal flip
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[1])                         # vertical flip
            x = torch.rot90(x, k=np.random.randint(4), dims=(1, 2))   # 0/90/180/270 degrees
            x = x * np.random.uniform(0.9, 1.1)                     # brightness (background stays 0)
        return x, torch.from_numpy(self.y[leaf]), torch.from_numpy(self.mask[leaf])


# --------------------------------------------------------------------------
# DATASET: one item = one random pixel from one training leaf (scripts 04, 08)
# --------------------------------------------------------------------------
class PixelDataset(Dataset):
    """Item i = (x (204,) float32 standardised spectrum, y (2,), mask (2,)) of one random pixel.

    One epoch draws `per_leaf` random pixels from every leaf in leaf_idx (a fresh draw every
    epoch), so every leaf counts equally whatever its size, and the pixel takes its leaf's
    labels. mean, std: (204,) from common.fit_band_standardiser on TRAIN leaves only.
    augment=True: +-10% brightness and Gaussian sensor noise (SD 0.05 after standardising).
    """

    def __init__(self, pixels, meta, leaf_idx, y, mask, mean, std, per_leaf, augment):
        self.pixels, self.meta = pixels, meta
        self.leaf_idx = np.asarray(leaf_idx)
        self.y, self.mask = y, mask
        self.mean, self.std = mean, std
        self.per_leaf = per_leaf          # pixels drawn from each leaf per epoch
        self.augment = augment
        self.px_start = meta.px_start.to_numpy()
        self.n_px = meta.n_px.to_numpy()

    def __len__(self):                    # how many examples make up one epoch
        return len(self.leaf_idx) * self.per_leaf

    def __getitem__(self, i):
        leaf = self.leaf_idx[i // self.per_leaf]              # which leaf this example belongs to
        # why not torch: np.random keeps the archived runs' random stream (seeded by seed_everything)
        px = self.pixels[self.px_start[leaf] + np.random.randint(self.n_px[leaf])]
        px = px.astype(np.float32)
        if self.augment:
            px = px * np.random.uniform(0.9, 1.1)              # brightness change (+-10%)
        x = (px - self.mean) / self.std                         # per-band standardisation
        if self.augment:
            x = x + np.random.normal(0, 0.05, x.shape).astype(np.float32)   # sensor noise
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(self.y[leaf]), torch.from_numpy(self.mask[leaf]))


# --------------------------------------------------------------------------
# PREDICTION: average over ALL pixels of each test leaf
# --------------------------------------------------------------------------
def predict_pixel_leaves(model, pixels, meta, leaf_idx, mean, std, device):
    """Leaf probabilities from a pixel model: (len(leaf_idx), 2) float64 NumPy array.

    Every pixel of each leaf is standardised with the TRAIN mean / std and scored in chunks of
    ~8192 pixels (GPU memory); the leaf's probability is the mean over its pixels.
    """
    # why not torch for the standardising / chunking: kept in NumPy on the CPU (cast to float32 for
    # the model) so the archived predictions stay bit-identical; the model itself runs on `device`
    out = np.zeros((len(leaf_idx), 2))
    for k, i in enumerate(leaf_idx):
        px = (np.asarray(leaf_pixels(pixels, meta, i)) - mean) / std
        probs = [predict_probs(model, torch.from_numpy(c.astype(np.float32)), device)
                 for c in np.array_split(px, max(1, len(px) // 8192))]
        out[k] = torch.cat(probs).mean(0).numpy()      # mean probability over pixels
    return out
