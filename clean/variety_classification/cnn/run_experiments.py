"""Train the variety CNN on held-out field rows, for one choice of input bands."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Needed by cuBLAS for deterministic matrix products; must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE.parent / "logistic_regression", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from cnn_common import augment, pad_to_canvas  # noqa: E402
from data import load_cubes, load_dataset  # noqa: E402
from log import get_logger  # noqa: E402
from logistic_regression import BANDS, rgb_bands  # noqa: E402
from model import VarietyCNN  # noqa: E402
from pca_common import VARIETIES, get_device, np_  # noqa: E402

RESULTS, LOGS = HERE / "results", HERE / "logs"
ROWS = ["R1", "R2", "R3", "R4", "R5"]
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Config:
    """What the network sees; SNV per pixel is always applied, over the bands kept."""

    bands: str = "all"    # "all": 204 bands; "rgb": the bands nearest 640 / 550 / 460 nm

    @property
    def name(self) -> str:
        """Folder name of this configuration."""
        return self.bands


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by every configuration; the same as the disease CNN, without jitter."""

    batch: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 80   # without validation, the exact number of epochs
    patience: int = 15     # epochs without a lower validation loss
    dropout: float = 0.5
    validation: bool = True  # False: train on all 4 non-test rows, no early stopping

    def suffix(self) -> str:
        """Folder-name suffix of a run without validation ("" with it)."""
        return "" if self.validation else f"_no_val_e{self.max_epochs}"


def load(device: torch.device) -> dict:
    """Put every leaf of every variety on one square canvas.

    n: the number of leaves (998). S: canvas side, set by the largest leaf.

    :param device: GPU or CPU.
    :type device: torch.device
    :return: meta, X (n, 204, S, S) float16, mask (n, 1, S, S), y (n,), band index of each
        RGB channel, canvas side.
    :rtype: dict
    """
    metas, cubes = [], []
    for variety in VARIETIES:
        m, c = load_cubes(variety)
        metas.append(m.assign(variety=variety))
        cubes += c
    meta = pd.concat(metas, ignore_index=True)
    all_meta, wavelengths, _means, _glint = load_dataset()
    # The report pairs these leaves with the logistic-regression leaves, so check the order.
    assert list(all_meta.path) == list(meta.path), "cube and mean-spectrum order differ"
    X, mask, side = pad_to_canvas(cubes, device)
    y = torch.tensor([VARIETIES.index(v) for v in meta.variety], device=device)
    return {"meta": meta, "X": X, "mask": mask, "y": y, "rgb": rgb_bands(wavelengths), "side": side}


def preprocess(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply SNV to each pixel over its own bands, which removes brightness; the background stays 0.

    B: batch size. d: input bands. S: canvas side.

    :param x: Reflectance, shape (B, d, S, S).
    :type x: torch.Tensor
    :param m: Leaf mask, shape (B, 1, S, S).
    :type m: torch.Tensor
    :return: Input, shape (B, d, S, S) float32.
    :rtype: torch.Tensor
    """
    x = x.float()
    # Background pixels have SD 0, so clamp and re-mask afterwards.
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6) * m


def fold_split(rows: np.ndarray, k: int, validation: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test = row k, validation (early stopping) = row k+1, train = the other 3 rows.

    Without validation the validation set is empty and row k+1 trains too.

    n: the number of leaves.

    :param rows: Field row per leaf, length n.
    :type rows: np.ndarray
    :param k: Fold number 0-4.
    :type k: int
    :param validation: Keep a validation row.
    :type validation: bool
    :return: train, validation, test leaf indices.
    :rtype: tuple
    """
    idx = np.arange(len(rows))
    test = rows == ROWS[k]
    val = rows == ROWS[(k + 1) % 5] if validation else np.zeros(len(rows), dtype=bool)
    return idx[~test & ~val], idx[val], idx[test]


def _logits(model, X, mask, idx, bands, batch: int = 16) -> torch.Tensor:
    """Logits for the given leaves, no augmentation (eval mode)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            out.append(model(preprocess(X[j][:, bands], mask[j]), mask[j]))
    return torch.cat(out)


def train_fold(d: dict, bands: torch.Tensor, tc: TrainConfig, split: tuple, seed: int) -> dict:
    """Train one network and score the test row.

    With validation, keep the weights with the lowest validation loss. Without it, train for
    exactly tc.max_epochs epochs and keep the last weights.

    k: the number of test leaves.

    :param d: Output of :func:`load`.
    :type d: dict
    :param bands: Indices of the input bands, length d.
    :type bands: torch.Tensor
    :param tc: Training settings.
    :type tc: TrainConfig
    :param split: (train, validation, test) index arrays from :func:`fold_split`.
    :type split: tuple
    :param seed: Seed for weights, batch order and augmentation.
    :type seed: int
    :return: test probabilities (k, 4), best epoch, epochs run, per-epoch train and validation losses.
    :rtype: dict
    """
    X, mask, y = d["X"], d["mask"], d["y"]
    dev = X.device
    tr, va, te = (torch.as_tensor(s, device=dev) for s in split)
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)

    model = VarietyCNN(len(bands), tc.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    best, hist_tr, hist_va = (float("inf"), 0, None), [], []
    for epoch in range(1, tc.max_epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g, device=dev)]
        total = 0.0
        for s in range(0, len(order), tc.batch):
            j = order[s:s + tc.batch]
            x, m = augment(preprocess(X[j][:, bands], mask[j]), mask[j], g)
            loss = F.cross_entropy(model(x, m), y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        hist_tr.append(total / len(tr))
        if not tc.validation:
            continue

        model.eval()
        vloss = F.cross_entropy(_logits(model, X, mask, va, bands), y[va]).item()
        hist_va.append(vloss)
        if vloss < best[0]:
            best = (vloss, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= tc.patience:
            break

    if tc.validation:
        model.load_state_dict(best[2])
    model.eval()
    prob = torch.softmax(_logits(model, X, mask, te, bands), 1)
    return {"prob": prob, "best_epoch": best[1] if tc.validation else len(hist_tr), "epochs": len(hist_tr), "train_loss": hist_tr, "val_loss": hist_va}


def invariance_check(d: dict, bands: torch.Tensor) -> float:
    """Check that padding and position don't change a leaf's score (untrained network, eval mode).

    :param d: Output of :func:`load`.
    :type d: dict
    :param bands: Indices of the input bands, length d.
    :type bands: torch.Tensor
    :return: Largest logit change from shifting a leaf by 8 px or enlarging the canvas by 32 px.
    :rtype: float
    """
    torch.manual_seed(0)
    m = d["mask"][:4]
    x = preprocess(d["X"][:4][:, bands], m)
    model = VarietyCNN(len(bands)).to(x.device).eval()
    for bn in model.bn:   # non-trivial running statistics, so eval mode is exercised
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
    with torch.no_grad():
        base = model(x, m)
        return max((model(F.pad(x, p), F.pad(m, p)) - base).abs().max().item()
                   for p in ((8, 0, 8, 0), (16, 16, 16, 16), (0, 32, 0, 32)))


def check_settings(root: Path, tc: TrainConfig) -> None:
    """Refuse to resume into a folder whose runs used other training settings.

    :param root: Results folder of this run.
    :type root: Path
    :param tc: Training settings of this run.
    :type tc: TrainConfig
    :raises RuntimeError: if root/config.json lists different training settings.
    """
    if (root / "config.json").exists():
        # A setting added after a run was made took its default value in that run.
        old = {**asdict(TrainConfig()), **json.loads((root / "config.json").read_text())["training"]}
        if old != asdict(tc):
            raise RuntimeError(f"{root} holds runs made with {old}, not {asdict(tc)}; move or delete it first")


def main() -> None:
    """Check, then train every (seed, test row) of one configuration not already on disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=BANDS, default=Config.bands, help="all 204 bands, or only the 3 RGB bands")
    ap.add_argument("--no-validation", action="store_true",
                    help="train on all 4 non-test rows, no early stopping; needs --epochs")
    ap.add_argument("--epochs", type=int, help="exact number of epochs of a --no-validation run")
    ap.add_argument("--smoke", action="store_true", help="first test row, seed 0, 3 epochs, into results/smoke/")
    args = ap.parse_args()
    if args.no_validation and args.epochs is None:
        ap.error("--no-validation needs --epochs: nothing else can say when to stop")

    cfg = Config(args.bands)
    tc = TrainConfig(max_epochs=3, patience=3) if args.smoke else TrainConfig()
    if args.no_validation:
        tc = replace(tc, max_epochs=3 if args.smoke else args.epochs, validation=False)
    seeds, folds = ((0,), (0,)) if args.smoke else (SEEDS, range(5))
    run_name = cfg.name + tc.suffix()
    root = (RESULTS / "smoke" if args.smoke else RESULTS) / run_name
    log = get_logger("variety_cnn", log_file=None if args.smoke else LOGS / f"{run_name}.log")
    check_settings(root, tc)

    torch.backends.cudnn.benchmark = False
    # warn_only: max-pool backward has no deterministic CUDA kernel; everything else is.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = get_device()

    d = load(device)
    bands = torch.as_tensor(d["rgb"] if cfg.bands == "rgb" else np.arange(d["X"].shape[1]), device=device)
    rows = d["meta"].row.to_numpy().astype(str)
    log.info("%s: %d leaves, %d bands, canvas %d", run_name, len(rows), len(bands), d["side"])
    change = invariance_check(d, bands)
    if change > 1e-3:
        raise RuntimeError(f"padding leaks into the prediction: {change}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({**asdict(cfg), "training": asdict(tc), "seeds": list(seeds),
                                                  "canvas": d["side"]}, indent=2))
    for seed in seeds:
        out = root / f"seed{seed}"
        out.mkdir(exist_ok=True)
        epochs, t_seed = [], time.time()
        for k in folds:
            f = out / f"fold{k}.npz"
            if f.exists():
                continue
            split = fold_split(rows, k, tc.validation)
            r = train_fold(d, bands, tc, split, 100 * seed + k)
            np.savez(f, test_idx=split[2], prob=np_(r["prob"]), best_epoch=r["best_epoch"], epochs=r["epochs"],
                     train_loss=np.array(r["train_loss"]), val_loss=np.array(r["val_loss"]))
            epochs.append(r["best_epoch"])
        if epochs:
            log.info("seed %d: %d folds, median best epoch %.0f, %.0f s", seed, len(epochs), np.median(epochs),
                     time.time() - t_seed)
    log.info("[done] %s", root.relative_to(HERE))


if __name__ == "__main__":
    main()
