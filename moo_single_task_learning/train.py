"""Train one six-head CNN on one row-held-out fold (LS or MGDA)."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from data import VineData, fold_split, make_batch
from metrics import score_all
from model import SixHeadCNN
from moo import (
    aggregate_mgda,
    aggregate_torchjd,
    collect_head_grads,
    head_losses,
    write_grad,
)


METHODS = ("ls", "mgda", "mgda_gn", "mgda_ln", "mgda_lgn", "svdpga", "osnmfpga")


@dataclass
class TrainConfig:
    """Training knobs (not tuned on the test row)."""

    method: str = "ls"       # "ls", "mgda", "mgda_gn", "mgda_ln", "mgda_lgn", or torchjd PGA: "svdpga", "osnmfpga"
    batch: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 40
    patience: int = 10       # stop if val loss has not improved
    seed: int = 0
    width: int = 32


def _epoch_indices(n: int, batch: int, g: torch.Generator):
    """Yield shuffled mini-batch index arrays for one epoch."""
    order = torch.randperm(n, generator=g).numpy()
    for start in range(0, n, batch):
        yield order[start:start + batch]


@torch.no_grad()
def eval_total_loss(
    model: nn.Module,
    data: VineData,
    indices: np.ndarray,
    device: torch.device,
    batch: int,
) -> float:
    """Mean sum-of-six-head-losses over indices (for early stopping)."""
    model.eval()
    total = 0.0
    n = 0
    for start in range(0, len(indices), batch):
        j = indices[start:start + batch]
        x, mask, y = make_batch(data, j, device)
        losses = head_losses(model(x, mask), y)
        total += sum(float(L.detach()) for L in losses) * len(j)
        n += len(j)
    return total / max(n, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    data: VineData,
    indices: np.ndarray,
    device: torch.device,
    batch: int,
) -> torch.Tensor:
    """Sigmoid probabilities (len(indices), 6) on CPU."""
    model.eval()
    out = []
    for start in range(0, len(indices), batch):
        j = indices[start:start + batch]
        x, mask, _y = make_batch(data, j, device)
        out.append(torch.sigmoid(model(x, mask)).cpu())
    return torch.cat(out, dim=0)


def train_fold(
    data: VineData,
    fold: int,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    """Train on fold ``fold``, score the held-out test row.

    Early stopping watches the sum of the six validation losses.
    Best weights are restored before the test pass.

    :return: metrics (variety + disease), best_epoch, method, fold.
    """
    assert cfg.method in METHODS, cfg.method
    norm = cfg.method.split("_")[1] if "_" in cfg.method else "none"   # MGDA gradient normalization
    tr, va, te = fold_split(data.fold_ids, fold)

    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)

    model = SixHeadCNN(width=cfg.width).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best = (float("inf"), 0, None)  # val_loss, epoch, state_dict
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for jb in _epoch_indices(len(tr), cfg.batch, g):
            x, mask, y = make_batch(data, tr[jb], device)
            losses = head_losses(model(x, mask), y)

            opt.zero_grad(set_to_none=True)
            if cfg.method == "ls":
                # Linear scalarization: one backward on the summed loss.
                sum(losses).backward()
            elif cfg.method.startswith("mgda"):
                # MGDA: six grads (optionally normalized) → shortest mix → write into .grad.
                loss_values = [float(L.detach()) for L in losses]
                grads = collect_head_grads(model, losses)
                g_agg, _alpha = aggregate_mgda(grads, loss_values, norm)
                write_grad(model, g_agg)
            else:
                # torchjd aggregator (PGA): six grads → its update direction → write into .grad.
                grads = collect_head_grads(model, losses)
                g_agg, _weights = aggregate_torchjd(grads, cfg.method)
                write_grad(model, g_agg)
            opt.step()

        vloss = eval_total_loss(model, data, va, device, cfg.batch)
        if vloss < best[0]:
            best = (vloss, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= cfg.patience:
            break

    model.load_state_dict(best[2])
    prob = predict(model, data, te, device, cfg.batch)
    y_te = torch.as_tensor(data.y[te])
    report = score_all(prob, y_te, data.varieties[te])

    return {
        "method": cfg.method,
        "fold": fold,
        "seed": cfg.seed,
        "best_epoch": best[1],
        "epochs_run": epoch,
        "val_loss": best[0],
        "metrics": report,
    }


if __name__ == "__main__":
    from data import get_device, load_vines

    device = get_device()
    print("loading vines on", device, "…")
    data = load_vines()

    # Smoke: 2 epochs, fold 0, LS only.
    cfg = TrainConfig(method="ls", max_epochs=2, patience=2, batch=4)
    result = train_fold(data, fold=0, cfg=cfg, device=device)

    m = result["metrics"]
    print(
        f"method={result['method']} fold={result['fold']} "
        f"best_epoch={result['best_epoch']} val_loss={result['val_loss']:.3f}",
    )
    print("variety", m["variety"])
    print("disease RB", m["disease"]["RedBlotch"])
    print("disease LR", m["disease"]["Leafroll"])