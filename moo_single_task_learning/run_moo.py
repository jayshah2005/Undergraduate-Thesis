"""Six-head CNN trained with each MOO aggregator, in the single-task CNNs' setup (../ANALYSIS_PLAN.md).

One network with six outputs (Baco, Cab Franc, Chardonnay, Vidal, red blotch, leafroll), one loss per
output, and the six gradients combined by a method from moo.py (LS, MGDA with or without
normalization, or a torchjd aggregator). Everything else is the disease CNN's: the small CNN or small
ResNet (two max pools, masked batch norm), gated attention pooling, dropout 0.5, the 212 px canvas,
the pixel normalization of the bands (SNV, or normalized rgb for RGB), rotations and flips, batch 8,
up to 80 epochs with patience 15, class-weighted binary cross-entropy (leafroll without Cab Franc),
the row folds and the seed rule 100 · seed + fold. The data loading, normalization and augmentation
are the disease code itself.

Optimizer: AdamW (lr 1e-3, weight decay 1e-2, as every single-task CNN) or SGD (momentum 0.9, weight
decay 5e-4, constant learning rate chosen for each method and network on the validation rows with
--select-lr, on 204 bands; RGB reuses that rate). Every aggregator works on the gradients in float64.

Checkpoints: test predictions are kept at four epochs, each the lowest validation loss of one choice
of heads: the 4 variety heads summed, red blotch, leafroll, and all six. The first three give per-task
scores that compare with the single-task models; the last is the single model. Training stops when
none of the four has improved for 15 epochs (or at 80).

Usage (from this folder; one job at a time: the 998 leaves take about 27 GB on the GPU)::

    ../.venv/bin/python -u run_moo.py --network small_resnet --optimizer adamw --method ls
    PYTHONPATH=../../torchjd/src ../.venv/bin/python -u run_moo.py --network small_cnn --method upgrad --bands rgb
    ../.venv/bin/python -u run_moo.py --network small_cnn --optimizer sgd --method ls --select-lr
    ../.venv/bin/python -u run_moo.py --network small_cnn --method mgda --smoke   # 1 fold, 2 epochs, results/smoke/
"""
from __future__ import annotations

import os

# Deterministic cuBLAS must be set before CUDA starts (as in the disease CNN runs).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
DISEASE_CNN = HERE.parent / "disease_classification" / "cnn"
# The disease folder has its own train.py and model.py: import from there, not from this folder.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(DISEASE_CNN))
import run_experiments as DE  # noqa: E402  (disease_classification/cnn; also puts the repo root on the path)
import train as DT  # noqa: E402
from helpers.backbones import BACKBONES, make_backbone, masked_norms  # noqa: E402
from helpers.cnn_common import LeafPool, augment  # noqa: E402
from logistic_regression import fold_of, fold_split, task_labels, usable_leaves  # noqa: E402

assert Path(DT.__file__).resolve().parent == DISEASE_CNN, f"wrong train module: {DT.__file__}"


def _load_moo():
    """This folder's moo.py (per-head gradients and the aggregators), loaded by path."""
    spec = importlib.util.spec_from_file_location("six_head_moo", HERE / "moo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moo = _load_moo()

RESULTS = HERE / "results"
METHODS = ("ls", "mgda", "mgda_gn", "mgda_ln", "mgda_lgn", "svdpga", "osnmfpga", "upgrad")
NETWORKS = ("small_cnn", "small_resnet")
OPTIMIZERS = ("adamw", "sgd")
SEEDS = DE.SEEDS
SGD_LR_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
SGD_MOMENTUM, SGD_WEIGHT_DECAY = 0.9, 5e-4   # as the MOO paper's classification runs
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]
HEADS = VARIETIES + ["red_blotch", "leafroll"]
# Checkpoint choices: which heads' validation losses are summed to pick the epoch.
CHECKPOINTS = {"variety": [0, 1, 2, 3], "red_blotch": [4], "leafroll": [5], "all": [0, 1, 2, 3, 4, 5]}
CODE_FILES = (HERE / "run_moo.py", HERE / "moo.py", DISEASE_CNN / "train.py", DISEASE_CNN / "run_experiments.py",
              HERE.parent / "helpers" / "backbones.py", HERE.parent / "helpers" / "cnn_common.py")


@dataclass(frozen=True)
class Config:
    """The disease CNN's settings (disease_classification/cnn/train.py TrainConfig), plus network, pooling and optimizer."""

    batch: int = DT.TrainConfig.batch
    lr: float = DT.TrainConfig.lr                       # AdamW
    weight_decay: float = DT.TrainConfig.weight_decay   # AdamW
    max_epochs: int = DT.TrainConfig.max_epochs
    patience: int = DT.TrainConfig.patience
    dropout: float = DT.TrainConfig.dropout
    augment: bool = DT.TrainConfig.augment
    backbone: str = "small_resnet"
    pooling: str = "attention"
    optimizer: str = "adamw"
    sgd_lr: float | None = None                         # SGD only; chosen on the validation rows


assert (Config.max_epochs, Config.patience, Config.dropout) == (80, 15, 0.5), "not the disease CNN's settings"


class SixHeadLeafCNN(nn.Module):
    """The disease LeafCNN with six outputs: backbone → masked pooling → dropout → 6 logits."""

    def __init__(self, n_bands: int, cfg: Config):
        super().__init__()
        self.backbone = make_backbone(cfg.backbone, n_bands)
        self.pool = LeafPool(self.backbone.out_channels, cfg.pooling)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(self.pool.out_dim, len(HEADS)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits (B, 6) for normalized leaves x (B, d, S, S) with leaf mask (B, 1, S, S)."""
        h, m = self.backbone(x, mask)
        z, _att = self.pool(h, m)
        return self.head(z)


def labels(meta) -> tuple[np.ndarray, np.ndarray]:
    """Six 0/1 targets per leaf and which of them count (leafroll leaves out Cab Franc, as the disease pool does).

    :return: y (n, 6) float32, valid (n, 6) float32.
    """
    n = len(meta)
    y = np.zeros((n, len(HEADS)), np.float32)
    valid = np.ones((n, len(HEADS)), np.float32)
    y[np.arange(n), meta.variety.map(VARIETIES.index).to_numpy()] = 1
    for col, task in ((4, "red_blotch"), (5, "leafroll")):
        t = task_labels(meta, task)
        y[:, col] = t
        valid[:, col] = usable_leaves(meta, t)
    return y, valid


def head_losses(logits, y, valid, pos_weight) -> list[torch.Tensor]:
    """Class-weighted binary cross-entropy per head, averaged over the leaves that count for it.

    A head with no counting leaf in the batch gets 0 (still part of the graph).
    """
    out = []
    for k in range(len(HEADS)):
        loss = F.binary_cross_entropy_with_logits(logits[:, k], y[:, k], pos_weight=pos_weight[k], reduction="none")
        out.append((loss * valid[:, k]).sum() / valid[:, k].sum().clamp_min(1.0))
    return out


def predict(model, X, mask, idx, norm: str, batch: int = 16) -> torch.Tensor:
    """Logits (len(idx), 6) in eval mode."""
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            out.append(model(DT.preprocess(X[j], mask[j], norm), mask[j]))
    return torch.cat(out)


def update(model, losses: list[torch.Tensor], present: list[int], method: str) -> torch.Tensor:
    """Backward pass(es) and the combined gradient written into .grad; returns the weight per head (6,).

    The aggregators get the six gradients in float64 (their Gram matrices and eigendecompositions are
    fragile in float32 when the gradient norms differ by many orders of magnitude).
    """
    weights = torch.zeros(len(HEADS))
    if method == "ls":
        sum(losses).backward()   # heads without a counting leaf add 0
        weights[present] = 1.0
        return weights
    head = [losses[k] for k in present]
    grads = [g.double() for g in moo.collect_head_grads(model, head)]
    if method.startswith("mgda"):
        norm = method.split("_")[1] if "_" in method else "none"
        g, w = moo.aggregate_mgda(grads, [float(L.detach()) for L in head], norm)
    else:
        g, w = moo.aggregate_torchjd(grads, method)
    moo.write_grad(model, g.float())
    weights[present] = w.detach().float().cpu()
    return weights


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW as the single-task CNNs, or SGD with momentum and the chosen learning rate."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    assert cfg.sgd_lr is not None, "SGD needs a learning rate (--select-lr first)"
    return torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)


def train_fold(d: dict, y: torch.Tensor, valid: torch.Tensor, split, seed: int, method: str, cfg: Config) -> dict:
    """Train one six-head CNN on one fold and keep the test predictions at each checkpoint choice.

    :param d: Output of the disease run_experiments.load (X, mask on the GPU, norm).
    :param y: Targets (n, 6). :param valid: Which targets count (n, 6).
    :param split: (train, validation, test) leaf indices.
    :param seed: Seed for weights, batch order and augmentation.
    :param method: A value of METHODS.
    :return: test probabilities and best validation loss per checkpoint choice, best epochs, per-epoch
        validation losses per head and weights per head.
    """
    X, mask, norm = d["X"], d["mask"], d["norm"]
    dev = X.device
    tr, va, te = (torch.as_tensor(s, device=dev) for s in split)
    torch.manual_seed(seed)
    g = torch.Generator(device=dev).manual_seed(seed)

    model = SixHeadLeafCNN(X.shape[1], cfg).to(dev)
    opt = make_optimizer(model, cfg)
    # Up-weight each head's positives by #neg / #pos among its counting training leaves.
    n_pos = (y[tr] * valid[tr]).sum(0)
    pos_weight = (valid[tr].sum(0) - n_pos) / n_pos

    best = {c: {"loss": float("inf"), "epoch": 0, "prob": None} for c in CHECKPOINTS}
    val_losses, epoch_weights = [], []
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g, device=dev)]
        step_weights = []
        for s in range(0, len(order), cfg.batch):
            j = order[s:s + cfg.batch]
            x, m = DT.preprocess(X[j], mask[j], norm), mask[j]
            if cfg.augment:
                x, m = augment(x, m, g)
            losses = head_losses(model(x, m), y[j], valid[j], pos_weight)
            present = [k for k in range(len(HEADS)) if valid[j, k].sum() > 0]
            opt.zero_grad(set_to_none=True)
            step_weights.append(update(model, losses, present, method))
            opt.step()
        epoch_weights.append(torch.stack(step_weights).mean(0))

        # Validation loss per head; keep the test predictions wherever a checkpoint choice improves.
        model.eval()
        lv = predict(model, X, mask, va, norm)
        per_head = torch.stack(head_losses(lv, y[va], valid[va], pos_weight)).cpu()
        val_losses.append(per_head)
        test_prob = None
        for c, heads in CHECKPOINTS.items():
            loss = float(per_head[heads].sum())
            if loss < best[c]["loss"]:
                if test_prob is None:
                    test_prob = torch.sigmoid(predict(model, X, mask, te, norm)).cpu()
                best[c] = {"loss": loss, "epoch": epoch, "prob": test_prob}
        if not np.isfinite(float(per_head.sum())):
            break   # diverged (possible with SGD at a too-large learning rate); the best epochs so far are kept
        if all(epoch - b["epoch"] >= cfg.patience for b in best.values()):
            break

    # A choice that never improved (only if training diverged from the first epoch) gets NaN predictions.
    out = {f"prob_{c}": (b["prob"].numpy() if b["prob"] is not None else np.full((len(te), len(HEADS)), np.nan))
           for c, b in best.items()}
    out.update({f"best_epoch_{c}": b["epoch"] for c, b in best.items()})
    out.update({f"val_loss_{c}": b["loss"] for c, b in best.items()})
    out.update(epochs_run=epoch, val_losses=torch.stack(val_losses).numpy(), weights=torch.stack(epoch_weights).numpy())
    return out


def invariance_check(n_bands: int, d: dict, cfg: Config) -> float:
    """Largest logit change when a leaf is shifted or put on a larger canvas (float64, eval mode, untrained)."""
    torch.manual_seed(0)
    X, mask = d["X"][:4].float(), d["mask"][:4]
    x = DT.preprocess(X, mask, d["norm"]).double()
    mask = mask.double()
    model = SixHeadLeafCNN(n_bands, cfg).to(X.device).eval()
    for bn in masked_norms(model):   # non-trivial running statistics, so eval mode is exercised
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2.0)
    model = model.double()
    k = BACKBONES[cfg.backbone].stride // 4
    with torch.no_grad():
        base = model(x, mask)
        return max((model(F.pad(x, p), F.pad(mask, p)) - base).abs().max().item()
                   for p in ((8 * k, 0, 8 * k, 0), (16 * k,) * 4, (0, 32 * k, 0, 32 * k)))


def code_hashes(method: str) -> dict[str, str]:
    """sha256 of every file the training depends on; the torchjd methods also record torchjd's commit."""
    hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE_FILES}
    if method in moo.TORCHJD_AGGREGATORS:
        import torchjd
        repo = Path(torchjd.__file__).resolve().parents[2]
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        hashes["torchjd"] = (f"{os.path.relpath(repo, HERE)} @ {git('rev-parse', 'HEAD')}"
                             + (" (uncommitted changes)" if git("status", "--porcelain") else ""))
    return hashes


def results_dir(network: str, optimizer: str, bands: str, smoke: bool = False) -> Path:
    """results/<network>_<optimizer>[_rgb] (under results/smoke/ for a smoke run)."""
    name = f"{network}_{optimizer}" + ("_rgb" if bands == "rgb" else "")
    return RESULTS / "smoke" / name if smoke else RESULTS / name


def check_config(out: Path, cfg: Config, method: str, norm: str) -> None:
    """Write out/config.json, or check it: shared settings must match; a method's code and SGD rate must not change.

    :raises SystemExit: if the settings differ from the saved runs.
    """
    path = out / "config.json"
    shared = {k: v for k, v in asdict(cfg).items() if k != "sgd_lr"}
    have = json.loads(path.read_text()) if path.exists() else {
        "train_config": shared, "pixel_norm": norm, "seed_rule": "100 * seed + fold", "seeds": list(SEEDS),
        "checkpoints": CHECKPOINTS, "created": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "by_method": {}}
    if have["train_config"] != shared or have["pixel_norm"] != norm:
        raise SystemExit(f"{path}: the training settings differ from the saved runs; use a new results folder.")
    entry = {"code_sha256": code_hashes(method), "sgd_lr": cfg.sgd_lr}
    if (out / method).exists() and have["by_method"].get(method, entry) != entry:
        raise SystemExit(f"{path}: the code or SGD rate changed since the saved '{method}' runs; use a new folder.")
    have["by_method"][method] = entry
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(have, indent=1) + "\n")


def setup(args) -> tuple[dict, torch.Tensor, torch.Tensor, np.ndarray]:
    """Deterministic GPU settings, the 998 leaves on the GPU (disease loader), the padding check, the targets."""
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)   # max-pool backward has no deterministic CUDA kernel
    device = DE.get_device()
    d = DE.load("all", args.bands, device, args.network)
    diff = invariance_check(d["X"].shape[1], d, Config(backbone=args.network))
    if diff > 1e-3:
        raise RuntimeError(f"padding changes the logits by {diff}")
    d["padding_check"] = diff
    y_np, valid_np = labels(d["meta"])
    y, valid = (torch.tensor(a, device=device) for a in (y_np, valid_np))
    return d, y, valid, fold_of(d["meta"])


def select_lr(args) -> None:
    """Choose one method's SGD learning rate on the validation rows: seed 0, every fold, each rate in SGD_LR_GRID.

    The score is the lowest summed validation loss of the six heads reached in training, averaged over
    the folds; the lowest wins. Chosen on 204 bands and reused for RGB; writes
    results/<network>_sgd/sgd_lr_<method>.json. The test rows are not used.
    """
    assert args.bands == "all", "SGD rates are chosen on 204 bands (RGB reuses them)"
    out = results_dir(args.network, "sgd", "all", args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    d, y, valid, fold_ids = setup(args)
    folds = range(1) if args.smoke else range(5)
    table = {}
    for lr in SGD_LR_GRID:
        cfg = Config(backbone=args.network, optimizer="sgd", sgd_lr=lr, **({"max_epochs": 2} if args.smoke else {}))
        losses = [train_fold(d, y, valid, fold_split(fold_ids, k), 100 * 0 + k, args.method, cfg)["val_loss_all"]
                  for k in folds]
        table[lr] = float(np.mean(losses))
        print(f"[{datetime.now():%F %T}] SGD lr {lr:g}: mean best validation loss {table[lr]:.4f}", flush=True)
    best = min(table, key=table.get)
    (out / f"sgd_lr_{args.method}.json").write_text(json.dumps(
        {"chosen_lr": best, "mean_best_val_loss": table, "grid": list(SGD_LR_GRID), "method": args.method,
         "momentum": SGD_MOMENTUM, "weight_decay": SGD_WEIGHT_DECAY}, indent=1) + "\n")
    print(f"[done] SGD lr {best:g} -> {out}/sgd_lr_{args.method}.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--network", choices=NETWORKS, required=True)
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--bands", choices=("all", "rgb"), default="all")
    ap.add_argument("--select-lr", action="store_true", help="choose SGD's learning rate on the validation rows")
    ap.add_argument("--sgd-lr-from", default="self",
                    help="whose sgd_lr_<method>.json to use: self (this method's own, the default) or another method")
    ap.add_argument("--smoke", action="store_true", help="1 seed, fold 0, 2 epochs, into results/smoke/")
    args = ap.parse_args()
    if args.select_lr:
        select_lr(args)
        return

    out = results_dir(args.network, args.optimizer, args.bands, args.smoke)
    sgd_lr = None
    if args.optimizer == "sgd":
        src = args.method if args.sgd_lr_from == "self" else args.sgd_lr_from
        lr_file = results_dir(args.network, "sgd", "all", args.smoke) / f"sgd_lr_{src}.json"   # chosen on 204 bands
        sgd_lr = json.loads(lr_file.read_text())["chosen_lr"]
    cfg = Config(backbone=args.network, optimizer=args.optimizer, sgd_lr=sgd_lr)
    seeds, folds = SEEDS, range(5)
    if args.smoke:
        cfg, seeds, folds = replace(cfg, max_epochs=2), (0,), range(1)
    d, y, valid, fold_ids = setup(args)
    check_config(out, cfg, args.method, d["norm"])
    meta, y_np, valid_np = d["meta"], y.cpu().numpy(), valid.cpu().numpy()
    log = open(out / f"{args.method}.log", "a")
    msg = (f"[{datetime.now():%F %T}] {out.name} {args.method}: {len(meta)} leaves, {d['X'].shape[1]} bands, "
           f"canvas {d['side']}, padding check {d['padding_check']:.1e}" + (f", SGD lr {sgd_lr:g}" if sgd_lr else ""))
    print(msg, flush=True)
    log.write(msg + "\n")
    for seed in seeds:
        for fold in folds:
            f = out / args.method / f"seed{seed}" / f"fold{fold}.npz"
            if f.exists():
                continue
            split = fold_split(fold_ids, fold)   # all 998 leaves; leafroll's Cab Franc leaves are masked in its loss
            t0 = time.time()
            r = train_fold(d, y, valid, split, 100 * seed + fold, args.method, cfg)
            seconds = time.time() - t0
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, test_idx=split[2], paths=np.array(list(meta.path), dtype=str)[split[2]], y=y_np[split[2]],
                     valid=valid_np[split[2]], seconds=seconds, **r)
            line = (f"[{datetime.now():%F %T}] {args.method} seed {seed} fold {fold}: best epochs "
                    + ", ".join(f"{c} {r[f'best_epoch_{c}']}" for c in CHECKPOINTS)
                    + f" of {r['epochs_run']}, {seconds:.0f} s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print(f"[done] {out}/{args.method}", flush=True)


if __name__ == "__main__":
    main()
