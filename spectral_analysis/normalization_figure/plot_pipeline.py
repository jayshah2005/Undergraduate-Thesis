"""Poster schematic: cube → 1×204 leaf mean → SNV → one row per vine.

The figure is a diagram, not data: a gray cube of small cubes, rows of square
cells, and a stack of rows. Only the rows are coloured, from a made-up
leaf-like spectrum, so the reader sees what averaging and SNV do to the values.
Everything is drawn in one set of axes where one cell is one unit wide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from helpers.diagram import (INK, INK_2, REFLECT, SNV_MAP, arrow, caption, key, mini_cube,  # noqa: E402
                     new_axes, row, save, voxel_cube)
from helpers.log import get_logger  # noqa: E402
from helpers.pca_common import np_, snv  # noqa: E402

FIG = HERE / "figures"
N_VINES = 998

# Diagram size: the cube is NX × NY pixels by NB band cells; rows show NB cells.
NX, NY, NB = 4, 4, 12
# The cube is drawn NB_CUBE slices deep (fewer than NB, so it stays shallow).
NB_CUBE = 8
DX, DY = 0.25, 0.18  # depth step per band slice, so the cube looks 3D

# Made-up leaf spectrum over NB cells: low visible, green bump, red edge, NIR plateau.
LEAF = [0.08, 0.10, 0.17, 0.11, 0.07, 0.06, 0.28, 0.52, 0.60, 0.62, 0.61, 0.59]
# Brightness of the stacked example vines before SNV removes it.
STACK_GAIN = [1.0, 0.7, 1.3, 0.85]

SNV_LIM = 1.6  # SNV values beyond ±SNV_LIM get the end colours


def compute(seed: int = 0) -> dict:
    """Build the made-up values the rows are coloured by.

    :param seed: Seed for the small vine-to-vine noise in the stack.
    :type seed: int
    :return: mean row (NB,), its SNV (NB,), SNV stack (len(STACK_GAIN), NB).
    :rtype: dict
    """
    g = torch.Generator().manual_seed(seed)
    leaf = torch.tensor(LEAF, dtype=torch.float64)
    # Each stacked vine is the leaf spectrum at its own brightness, plus a little noise.
    stack = torch.stack([k * leaf + 0.02 * torch.randn(NB, generator=g, dtype=torch.float64)
                         for k in STACK_GAIN])
    return {
        "mean": np_(leaf),
        "snv_row": np_(snv(leaf.unsqueeze(0))[0]),
        "stack_snv": np_(snv(stack)),
    }


def fig_pipeline(r: dict) -> list[Path]:
    """Plot the four-step schematic and save it as PNG and PDF.

    :param r: Output of :func:`compute`.
    :type r: dict
    :return: Paths written.
    :rtype: list[Path]
    """
    fig, ax = new_axes(12, 2.9)
    y_row = 2.0          # bottom of the single rows, level with the cube's middle
    step_gap = 7.0       # room between steps for the arrow and its label
    y_cap = -2.0         # one baseline for every caption
    vmax = max(LEAF) * 1.1

    # Step 1: the cube, with its three axes labelled.
    cw, _ch = voxel_cube(ax, 0.0, 0.0, NX, NY, NB_CUBE, DX, DY)
    ax.text(NX / 2, -0.35, "x", ha="center", va="top", fontsize=9.5, color=INK_2,
            style="italic")
    ax.text(-0.35, NY / 2, "y", ha="right", va="center", fontsize=9.5, color=INK_2,
            style="italic")
    ax.text(NX + NB_CUBE * DX / 2 + 0.2, NB_CUBE * DY / 2 - 0.3, "λ (204 bands)", ha="left",
            va="top", fontsize=9, color=INK_2)
    caption(ax, cw / 2, y_cap, "Hyperspectral cube")

    # Step 2: average every pixel, band by band → one 1 × 204 row.
    x2 = cw + step_gap
    arrow(ax, cw + 0.7, x2 - 0.7, y_row + 0.5, "average\nall pixels")
    row(ax, x2, y_row, r["mean"], REFLECT, 0.0, vmax)
    key(ax, x2 + NB / 2, y_row - 0.95, REFLECT, ("low", "", "high"))
    ax.text(x2 + NB, y_row + 1.3, "1 × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    caption(ax, x2 + NB / 2, y_cap, "Average spectrum")

    # Step 3: SNV the row → still 1 × 204.
    x3 = x2 + NB + step_gap
    arrow(ax, x2 + NB + 0.7, x3 - 0.7, y_row + 0.5, "SNV")
    row(ax, x3, y_row, r["snv_row"], SNV_MAP, -SNV_LIM, SNV_LIM)
    key(ax, x3 + NB / 2, y_row - 0.95, SNV_MAP, ("−", "0", "+"))
    ax.text(x3 + NB, y_row + 1.3, "1 × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    caption(ax, x3 + NB / 2, y_cap, "Normalised spectrum")

    # Step 4: repeat for every cube; one row each.
    x4 = x3 + NB + step_gap
    arrow(ax, x3 + NB + 0.7, x4 - 0.7, y_row + 0.5, "repeat for\nevery cube")
    xc = x4 + 2.6        # left edge of the cube icons; numbers sit to their left
    xr = xc + 4.0        # left edge of the stacked rows
    gap, dots = 0.35, 1.2
    ys = [dots + 2 * gap + 3 - i * (1 + gap) for i in range(3)] + [0.0]
    for y, lab, values in zip(ys, ["1", "2", "3", f"{N_VINES}"], r["stack_snv"]):
        ax.text(xc - 0.3, y + 0.5, lab, ha="right", va="center", fontsize=9, color=INK_2)
        mini_cube(ax, xc, y + 0.12, 0.7)
        arrow(ax, xc + 1.6, xr - 0.4, y + 0.5, size="small")
        row(ax, xr, y, values, SNV_MAP, -SNV_LIM, SNV_LIM)
    for x in (xc + 0.45, xr + NB / 2):
        ax.text(x, 1 + dots / 2, "⋮", ha="center", va="center", fontsize=11, color=INK_2)
    ax.text(xr + NB, ys[0] + 1.3, f"{N_VINES} × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    caption(ax, (xc + xr + NB) / 2, y_cap, "All cubes stacked")

    # imshow (the colour keys) resets the aspect; put it back so cells stay square.
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, xr + NB + 0.3)
    ax.set_ylim(-3.1, ys[0] + 2.4)

    paths = [FIG / "pipeline.png", FIG / "pipeline.pdf"]
    save(fig, paths[0])  # PDF (paths[1]) not written; pass it to save() to get one
    return paths


def main() -> None:
    """Write figures/pipeline.png (the PDF line in fig_pipeline is commented out)."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_pipeline")
    log.info("[saved] figures/%s", ", ".join(p.name for p in fig_pipeline(compute()) if p.exists()))


if __name__ == "__main__":
    main()
