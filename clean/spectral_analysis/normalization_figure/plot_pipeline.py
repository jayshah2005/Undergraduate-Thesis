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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "PCA"))
from log import get_logger  # noqa: E402
from pca_common import np_, snv  # noqa: E402

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

# The cube is ink and light grays (top and side shaded so it reads as 3D).
INK, INK_2 = "#111111", "#555555"
FRONT, TOP, SIDE = "white", "#eeeeee", "#d6d6d6"
LW_CELL, LW_EDGE = 0.7, 1.4
# Rows: one-hue green ramp for reflectance (light = low); blue ↔ gray ↔ red for
# SNV, centred on 0 so blue is below the vine's own mean and red above it.
REFLECT = LinearSegmentedColormap.from_list(
    "reflect", ["#eef7f1", "#a8dcbc", "#4fb37f", "#1d7f52", "#0b4d31"])
SNV_MAP = LinearSegmentedColormap.from_list(
    "snv", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])
SNV_LIM = 1.6


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


def _quad(ax, pts, face: str, lw: float = LW_CELL) -> None:
    """Draw one cell (square or parallelogram).

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param pts: Four corner points.
    :type pts: list
    :param face: Fill colour.
    :type face: str
    :param lw: Edge line width.
    :type lw: float
    """
    ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=INK, lw=lw,
                         joinstyle="round"))


def _voxel_cube(ax, x0: float, y0: float) -> tuple[float, float]:
    """Draw a cube made of small cubes: NX × NY pixels, NB_CUBE band slices deep.

    Only the three visible faces are drawn.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Front-face lower-left x.
    :type x0: float
    :param y0: Front-face lower-left y.
    :type y0: float
    :return: Total width and height of the drawing.
    :rtype: tuple[float, float]
    """
    for j in range(NY):  # front face: one cell per pixel
        for i in range(NX):
            bx, by = x0 + i, y0 + j
            _quad(ax, [(bx, by), (bx + 1, by), (bx + 1, by + 1), (bx, by + 1)], FRONT)
    for k in range(NB_CUBE):  # top face: pixels across, bands going back
        for i in range(NX):
            bx, by = x0 + i + k * DX, y0 + NY + k * DY
            _quad(ax, [(bx, by), (bx + 1, by), (bx + 1 + DX, by + DY), (bx + DX, by + DY)], TOP)
    for k in range(NB_CUBE):  # side face: pixels up, bands going back
        for j in range(NY):
            bx, by = x0 + NX + k * DX, y0 + j + k * DY
            _quad(ax, [(bx, by), (bx + DX, by + DY), (bx + DX, by + DY + 1), (bx, by + 1)], SIDE)
    Dx, Dy = NB_CUBE * DX, NB_CUBE * DY
    outline = [(x0, y0), (x0 + NX, y0), (x0 + NX + Dx, y0 + Dy), (x0 + NX + Dx, y0 + NY + Dy),
               (x0 + Dx, y0 + NY + Dy), (x0, y0 + NY)]
    ax.add_patch(Polygon(outline, closed=True, facecolor="none", edgecolor=INK, lw=LW_EDGE,
                         joinstyle="round"))
    ax.plot([x0, x0 + NX, x0 + NX + Dx], [y0 + NY, y0 + NY, y0 + NY + Dy], color=INK,
            lw=LW_EDGE, solid_joinstyle="round")
    ax.plot([x0 + NX, x0 + NX], [y0, y0 + NY], color=INK, lw=LW_EDGE)
    return NX + Dx, NY + Dy


def _mini_cube(ax, x0: float, y0: float, s: float) -> None:
    """Draw a small solid cube icon.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Front-face lower-left x.
    :type x0: float
    :param y0: Front-face lower-left y.
    :type y0: float
    :param s: Front-face size.
    :type s: float
    """
    dx, dy, lw = 0.55 * s, 0.38 * s, LW_EDGE * 0.8
    _quad(ax, [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)], FRONT, lw)
    _quad(ax, [(x0, y0 + s), (x0 + s, y0 + s), (x0 + s + dx, y0 + s + dy), (x0 + dx, y0 + s + dy)],
          TOP, lw)
    _quad(ax, [(x0 + s, y0), (x0 + s + dx, y0 + dy), (x0 + s + dx, y0 + s + dy), (x0 + s, y0 + s)],
          SIDE, lw)


def _row(ax, x0: float, y0: float, values, cmap, vmin: float, vmax: float) -> None:
    """Draw one 1 × NB row of square cells, coloured by value.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Lower-left x.
    :type x0: float
    :param y0: Lower-left y.
    :type y0: float
    :param values: One value per cell, shape (NB,).
    :type values: np.ndarray
    :param cmap: Colormap.
    :type cmap: matplotlib.colors.Colormap
    :param vmin: Value at the low end of cmap.
    :type vmin: float
    :param vmax: Value at the high end of cmap.
    :type vmax: float
    """
    for k, v in enumerate(values):
        bx = x0 + k
        face = cmap(float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)))
        _quad(ax, [(bx, y0), (bx + 1, y0), (bx + 1, y0 + 1), (bx, y0 + 1)], face)
    ax.add_patch(Rectangle((x0, y0), NB, 1, facecolor="none", edgecolor=INK, lw=LW_EDGE))


def _arrow(ax, x0: float, x1: float, y: float, label: str = "", size: str = "big") -> None:
    """Draw a right-pointing arrow with an optional label above it.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Tail x.
    :type x0: float
    :param x1: Head x.
    :type x1: float
    :param y: Arrow y.
    :type y: float
    :param label: Text above the arrow.
    :type label: str
    :param size: "big" between steps, "small" inside the stack.
    :type size: str
    """
    lw, scale = (1.6, 16) if size == "big" else (0.9, 9)
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=lw, mutation_scale=scale,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y + 0.45, label, ha="center", va="bottom", fontsize=9,
                color=INK_2, linespacing=1.15)


def _key(ax, xc: float, y: float, cmap, labels: tuple) -> None:
    """Draw a small colour key centred at xc, with labels under its ends (and middle).

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param xc: Centre x.
    :type xc: float
    :param y: Bottom of the bar.
    :type y: float
    :param cmap: Colormap.
    :type cmap: matplotlib.colors.Colormap
    :param labels: (left, middle or "", right) text.
    :type labels: tuple
    """
    w, h = 4.0, 0.35
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect="auto",
              extent=(xc - w / 2, xc + w / 2, y, y + h))
    ax.add_patch(Rectangle((xc - w / 2, y), w, h, facecolor="none", edgecolor=INK_2, lw=0.6))
    for x, text in zip((xc - w / 2, xc, xc + w / 2), labels):
        if text:
            ax.text(x, y - 0.12, text, ha="center", va="top", fontsize=7.5, color=INK_2)


def _caption(ax, x: float, y: float, text: str) -> None:
    """Write a small caption centred under a diagram.

    :param ax: Axes to write on.
    :type ax: matplotlib.axes.Axes
    :param x: Centre x.
    :type x: float
    :param y: Top of the text.
    :type y: float
    :param text: Caption.
    :type text: str
    """
    ax.text(x, y, text, ha="center", va="top", fontsize=9.5, color=INK)


def fig_pipeline(r: dict) -> list[Path]:
    """Plot the four-step schematic and save it as PNG and PDF.

    :param r: Output of :func:`compute`.
    :type r: dict
    :return: Paths written.
    :rtype: list[Path]
    """
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(12, 2.9), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    y_row = 2.0          # bottom of the single rows, level with the cube's middle
    step_gap = 7.0       # room between steps for the arrow and its label
    y_cap = -2.0         # one baseline for every caption
    vmax = max(LEAF) * 1.1

    # Step 1: the cube, with its three axes labelled.
    cw, _ch = _voxel_cube(ax, 0.0, 0.0)
    ax.text(NX / 2, -0.35, "x", ha="center", va="top", fontsize=9.5, color=INK_2,
            style="italic")
    ax.text(-0.35, NY / 2, "y", ha="right", va="center", fontsize=9.5, color=INK_2,
            style="italic")
    ax.text(NX + NB_CUBE * DX / 2 + 0.2, NB_CUBE * DY / 2 - 0.3, "λ (204 bands)", ha="left",
            va="top", fontsize=9, color=INK_2)
    _caption(ax, cw / 2, y_cap, "Hyperspectral cube")

    # Step 2: average every pixel, band by band → one 1 × 204 row.
    x2 = cw + step_gap
    _arrow(ax, cw + 0.7, x2 - 0.7, y_row + 0.5, "average\nall pixels")
    _row(ax, x2, y_row, r["mean"], REFLECT, 0.0, vmax)
    _key(ax, x2 + NB / 2, y_row - 0.95, REFLECT, ("low", "", "high"))
    ax.text(x2 + NB, y_row + 1.3, "1 × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    _caption(ax, x2 + NB / 2, y_cap, "Average spectrum")

    # Step 3: SNV the row → still 1 × 204.
    x3 = x2 + NB + step_gap
    _arrow(ax, x2 + NB + 0.7, x3 - 0.7, y_row + 0.5, "SNV")
    _row(ax, x3, y_row, r["snv_row"], SNV_MAP, -SNV_LIM, SNV_LIM)
    _key(ax, x3 + NB / 2, y_row - 0.95, SNV_MAP, ("−", "0", "+"))
    ax.text(x3 + NB, y_row + 1.3, "1 × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    _caption(ax, x3 + NB / 2, y_cap, "Normalised spectrum")

    # Step 4: repeat for every cube; one row each.
    x4 = x3 + NB + step_gap
    _arrow(ax, x3 + NB + 0.7, x4 - 0.7, y_row + 0.5, "repeat for\nevery cube")
    xc = x4 + 2.6        # left edge of the cube icons; numbers sit to their left
    xr = xc + 4.0        # left edge of the stacked rows
    gap, dots = 0.35, 1.2
    ys = [dots + 2 * gap + 3 - i * (1 + gap) for i in range(3)] + [0.0]
    for y, lab, row in zip(ys, ["1", "2", "3", f"{N_VINES}"], r["stack_snv"]):
        ax.text(xc - 0.3, y + 0.5, lab, ha="right", va="center", fontsize=9, color=INK_2)
        _mini_cube(ax, xc, y + 0.12, 0.7)
        _arrow(ax, xc + 1.6, xr - 0.4, y + 0.5, size="small")
        _row(ax, xr, y, row, SNV_MAP, -SNV_LIM, SNV_LIM)
    for x in (xc + 0.45, xr + NB / 2):
        ax.text(x, 1 + dots / 2, "⋮", ha="center", va="center", fontsize=11, color=INK_2)
    ax.text(xr + NB, ys[0] + 1.3, f"{N_VINES} × 204", ha="right", va="bottom", fontsize=9.5,
            color=INK, fontweight="bold")
    _caption(ax, (xc + xr + NB) / 2, y_cap, "All cubes stacked")

    # imshow (the colour keys) resets the aspect; put it back so cells stay square.
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, xr + NB + 0.3)
    ax.set_ylim(-3.1, ys[0] + 2.4)

    paths = [FIG / "pipeline.png", FIG / "pipeline.pdf"]
    fig.savefig(paths[0], dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    # fig.savefig(paths[1], facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return paths


def main() -> None:
    """Write figures/pipeline.png (the PDF line in fig_pipeline is commented out)."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_pipeline")
    log.info("[saved] figures/%s", ", ".join(p.name for p in fig_pipeline(compute()) if p.exists()))


if __name__ == "__main__":
    main()
