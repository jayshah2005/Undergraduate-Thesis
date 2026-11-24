"""Per-head losses and gradient aggregation (LS and MGDA)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HEADS = 6  # redefined; same meaning as data/model


def head_losses(logits: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """One binary CE-with-logits per head.

    BCE-with-logits = stable binary cross-entropy on raw scores (no sigmoid first).
    """
    return [
        F.binary_cross_entropy_with_logits(logits[:, m], y[:, m])
        for m in range(N_HEADS)
    ]


def flat_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate every parameter's .grad into one long vector."""
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


def write_grad(model: nn.Module, g_flat: torch.Tensor) -> None:
    """Copy a flat vector back into each p.grad (what opt.step reads)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = g_flat[offset:offset + n].view_as(p).clone()
        offset += n


def collect_head_grads(
    model: nn.Module,
    losses: list[torch.Tensor],
) -> list[torch.Tensor]:
    """One backward per head; return M flat gradients.

    retain_graph keeps the forward tape until the last head.
    """
    grads = []
    for m, loss_m in enumerate(losses):
        model.zero_grad(set_to_none=True)
        loss_m.backward(retain_graph=(m < len(losses) - 1))
        grads.append(flat_grads(model))
    return grads


def aggregate_ls(grads: list[torch.Tensor]) -> torch.Tensor:
    """Linear scalarization in gradient space: sum the objective grads.

    Same direction as sum(losses).backward() (up to float noise).
    """
    return sum(grads)


def mgda_alphas(
    grads: list[torch.Tensor],
    max_iter: int = 25,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Simplex weights α minimizing ||Σ α_m g_m||² (Frank–Wolfe on the Gram matrix).

    α ≥ 0, sum α = 1. Shortest mix of the six arrows = common descent (or stuck).
    """
    g = torch.stack(grads)   # (M, P)
    gg = g @ g.T             # (M, M); gg[i,j] = g_i · g_j
    m = g.shape[0]
    alpha = torch.full((m,), 1.0 / m, device=g.device, dtype=g.dtype)

    for _ in range(max_iter):
        t = int(torch.argmin(gg @ alpha).item())
        e = torch.zeros_like(alpha)
        e[t] = 1.0

        a_gg_a = alpha @ gg @ alpha
        e_gg_e = gg[t, t]
        e_gg_a = gg[t] @ alpha
        denom = a_gg_a + e_gg_e - 2 * e_gg_a
        if float(denom) < eps:
            break
        eta = torch.clamp((a_gg_a - e_gg_a) / denom, 0.0, 1.0)
        alpha = (1.0 - eta) * alpha + eta * e
        if float(eta) < eps:
            break
    return alpha


MGDA_NORMS = ("none", "gn", "ln", "lgn")


def normalize_grads(
    grads: list[torch.Tensor],
    losses: list[float] | None,
    norm: str,
) -> list[torch.Tensor]:
    """Rescale each objective's gradient before MGDA (normalization variants listed in LibMTL, Lin & Zhang 2023).

    "gn": g / ||g||, "ln": g / L, "lgn": g / (L ||g||), "none": unchanged.
    L is the head's loss on the current mini-batch.
    """
    assert norm in MGDA_NORMS, norm
    if norm == "none":
        return grads
    scale = [torch.ones((), device=g.device, dtype=g.dtype) for g in grads]
    if norm in ("gn", "lgn"):
        scale = [s * g.norm() for s, g in zip(scale, grads)]
    if norm in ("ln", "lgn"):
        assert losses is not None, "loss normalization needs the head losses"
        scale = [s * L for s, L in zip(scale, losses)]
    return [g / s.clamp_min(1e-12) for g, s in zip(grads, scale)]


def aggregate_mgda(
    grads: list[torch.Tensor],
    losses: list[float] | None = None,
    norm: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """MGDA update direction and the α that produced it.

    With a normalization, α solves the min-norm problem on the rescaled gradients and the
    update mixes those rescaled gradients: G = Σ α_m g̃_m.
    """
    grads = normalize_grads(grads, losses, norm)
    alpha = mgda_alphas(grads)
    g_mgda = sum(alpha[m] * grads[m] for m in range(len(grads)))
    return g_mgda, alpha


TORCHJD_AGGREGATORS = ("svdpga", "osnmfpga", "upgrad")


def aggregate_torchjd(grads: list[torch.Tensor], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Update direction from a torchjd aggregator, and the weight it gave each objective.

    Uses the local torchjd (../../torchjd/src on PYTHONPATH), with default settings:
    SVDPGA and OSNMFPGA (Principal Gradient Alignment: the average of the oriented principal
    directions of the gradients, or of the basis of an orthogonal semi-nonnegative factorization)
    and UPGrad (Quinton & Rey 2024, "Jacobian Descent for Multi-Objective Optimization": each
    gradient projected onto the dual cone of all of them, then averaged; needs quadprog and
    qpsolvers). The weights are only recorded; the direction is the aggregator's own.
    """
    from torchjd.aggregation import OSNMFPGA, SVDPGA, UPGrad  # only these methods need torchjd

    aggregator = {"svdpga": SVDPGA, "osnmfpga": OSNMFPGA, "upgrad": UPGrad}[name]()
    J = torch.stack(grads)   # (M, P): one row per objective
    weights = aggregator.gramian_weighting(J @ J.T)
    return aggregator(J), weights


if __name__ == "__main__":
    from data import HEAD_NAMES, fold_split, get_device, load_vines, make_batch
    from model import SixHeadCNN

    device = get_device()
    data = load_vines()
    _tr, _va, te = fold_split(data.fold_ids, 0)
    x, mask, y = make_batch(data, te[:8], device)

    model = SixHeadCNN().to(device)
    losses = head_losses(model(x, mask), y)

    # Path A: ordinary LS
    model.zero_grad(set_to_none=True)
    sum(losses).backward()
    g_ls = flat_grads(model)

    # Path B: per-head grads (new forward so graphs stay independent)
    model.zero_grad(set_to_none=True)
    losses = head_losses(model(x, mask), y)
    grads = collect_head_grads(model, losses)

    g_sum = aggregate_ls(grads)
    g_mgda, alpha = aggregate_mgda(grads)

    print("max |g_sum - g_ls|:", (g_sum - g_ls).abs().max().item())
    print(
        "||g_ls/6||", (g_ls / 6).norm().item(),
        "||g_mgda||", g_mgda.norm().item(),
    )
    print("α:", {n: round(a, 3) for n, a in zip(HEAD_NAMES, alpha.tolist())})