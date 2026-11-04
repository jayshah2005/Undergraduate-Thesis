"""Drawing pieces shared by the poster schematics (pipeline, CNN architectures).

Every schematic draws in one set of matplotlib axes where one cell is one unit wide,
with the same ink, grays, arrows and caption style, so the figures match on a poster.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np

FONT = ["Noto Sans", "DejaVu Sans"]
INK, INK_2 = "#111111", "#555555"
FRONT, TOP, SIDE = "white", "#eeeeee", "#d6d6d6"
LW_CELL, LW_EDGE = 0.7, 1.4
# One-hue green ramp for reflectance (light = low); blue ↔ gray ↔ red for SNV,
# centred on 0 so blue is below the spectrum's own mean and red above it.
REFLECT = LinearSegmentedColormap.from_list(
    "reflect", ["#eef7f1", "#a8dcbc", "#4fb37f", "#1d7f52", "#0b4d31"])
SNV_MAP = LinearSegmentedColormap.from_list(
    "snv", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])


def new_axes(width: float, height: float):
    """Open a white figure with one equal-aspect axes and no frame.

    :param width: Figure width in inches.
    :type width: float
    :param height: Figure height in inches.
    :type height: float
    :return: (figure, axes).
    :rtype: tuple
    """
    plt.rcParams.update({"font.family": FONT})
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, path) -> None:
    """Save at 300 dpi with a small white margin, then close the figure.

    :param fig: Figure to save.
    :type fig: matplotlib.figure.Figure
    :param path: PNG path.
    :type path: pathlib.Path
    """
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def shade(color: str, f: float) -> tuple:
    """Mix a colour with white (f > 0) or black (f < 0), for the faces of a box.

    :param color: Base colour.
    :type color: str
    :param f: Amount, -1 (black) to 1 (white).
    :type f: float
    :return: RGB tuple.
    :rtype: tuple
    """
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * f) if f >= 0 else tuple(c * (1 + f))


def quad(ax, pts, face, lw: float = LW_CELL) -> None:
    """Draw one cell (square or parallelogram).

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param pts: Four corner points.
    :type pts: list
    :param face: Fill colour.
    :type face: str | tuple
    :param lw: Edge line width.
    :type lw: float
    """
    ax.add_patch(Polygon(pts, closed=True, facecolor=face, edgecolor=INK, lw=lw,
                         joinstyle="round"))


def voxel_cube(ax, x0: float, y0: float, nx: int, ny: int, nb: int, dx: float, dy: float,
               front=None) -> tuple[float, float]:
    """Draw a cube made of small cubes: nx × ny pixels, nb band slices deep.

    Only the three visible faces are drawn.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Front-face lower-left x.
    :type x0: float
    :param y0: Front-face lower-left y.
    :type y0: float
    :param nx: Pixels across.
    :type nx: int
    :param ny: Pixels up.
    :type ny: int
    :param nb: Band slices drawn going back.
    :type nb: int
    :param dx: Depth step per slice in x.
    :type dx: float
    :param dy: Depth step per slice in y.
    :type dy: float
    :param front: Optional colour per front cell, indexed [row j][column i]; white if None.
    :type front: list | None
    :return: Total width and height of the drawing.
    :rtype: tuple[float, float]
    """
    for j in range(ny):  # front face: one cell per pixel
        for i in range(nx):
            bx, by = x0 + i, y0 + j
            face = FRONT if front is None else front[j][i]
            quad(ax, [(bx, by), (bx + 1, by), (bx + 1, by + 1), (bx, by + 1)], face)
    for k in range(nb):  # top face: pixels across, bands going back
        for i in range(nx):
            bx, by = x0 + i + k * dx, y0 + ny + k * dy
            quad(ax, [(bx, by), (bx + 1, by), (bx + 1 + dx, by + dy), (bx + dx, by + dy)], TOP)
    for k in range(nb):  # side face: pixels up, bands going back
        for j in range(ny):
            bx, by = x0 + nx + k * dx, y0 + j + k * dy
            quad(ax, [(bx, by), (bx + dx, by + dy), (bx + dx, by + dy + 1), (bx, by + 1)], SIDE)
    Dx, Dy = nb * dx, nb * dy
    outline = [(x0, y0), (x0 + nx, y0), (x0 + nx + Dx, y0 + Dy), (x0 + nx + Dx, y0 + ny + Dy),
               (x0 + Dx, y0 + ny + Dy), (x0, y0 + ny)]
    ax.add_patch(Polygon(outline, closed=True, facecolor="none", edgecolor=INK, lw=LW_EDGE,
                         joinstyle="round"))
    ax.plot([x0, x0 + nx, x0 + nx + Dx], [y0 + ny, y0 + ny, y0 + ny + Dy], color=INK,
            lw=LW_EDGE, solid_joinstyle="round")
    ax.plot([x0 + nx, x0 + nx], [y0, y0 + ny], color=INK, lw=LW_EDGE)
    return nx + Dx, ny + Dy


def mini_cube(ax, x0: float, y0: float, s: float) -> None:
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
    quad(ax, [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)], FRONT, lw)
    quad(ax, [(x0, y0 + s), (x0 + s, y0 + s), (x0 + s + dx, y0 + s + dy), (x0 + dx, y0 + s + dy)],
         TOP, lw)
    quad(ax, [(x0 + s, y0), (x0 + s + dx, y0 + dy), (x0 + s + dx, y0 + s + dy), (x0 + s, y0 + s)],
         SIDE, lw)


def box(ax, x0: float, yc: float, w: float, h: float, d: float, color: str,
        ox: float = 0.30, oy: float = 0.22) -> None:
    """Draw a solid box (a feature map): w wide, h tall, d deep, centred on yc.

    Front face lighter-than-side, top lightest, so the box reads as 3D.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Front-face left x.
    :type x0: float
    :param yc: Front-face centre y.
    :type yc: float
    :param w: Front-face width (channels).
    :type w: float
    :param h: Front-face height (map size).
    :type h: float
    :param d: Depth (map size), drawn going back and up.
    :type d: float
    :param color: Base colour.
    :type color: str
    :param ox: Depth step in x per unit of d.
    :type ox: float
    :param oy: Depth step in y per unit of d.
    :type oy: float
    """
    y0, Dx, Dy = yc - h / 2, d * ox, d * oy
    quad(ax, [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], shade(color, 0.15), LW_EDGE)
    quad(ax, [(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + Dx, y0 + h + Dy), (x0 + Dx, y0 + h + Dy)],
         shade(color, 0.5), LW_EDGE)
    quad(ax, [(x0 + w, y0), (x0 + w + Dx, y0 + Dy), (x0 + w + Dx, y0 + h + Dy), (x0 + w, y0 + h)],
         shade(color, -0.12), LW_EDGE)


def row(ax, x0: float, y0: float, values, cmap, vmin: float, vmax: float) -> None:
    """Draw one 1 × n row of square cells, coloured by value.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Lower-left x.
    :type x0: float
    :param y0: Lower-left y.
    :type y0: float
    :param values: One value per cell, shape (n,).
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
        quad(ax, [(bx, y0), (bx + 1, y0), (bx + 1, y0 + 1), (bx, y0 + 1)], face)
    ax.add_patch(Rectangle((x0, y0), len(values), 1, facecolor="none", edgecolor=INK, lw=LW_EDGE))


def column(ax, x0: float, yc: float, colors, s: float = 0.55) -> tuple[float, float]:
    """Draw an n × 1 column of square cells (a vector), centred on yc.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Left x.
    :type x0: float
    :param yc: Centre y.
    :type yc: float
    :param colors: One colour per cell, top first.
    :type colors: list
    :param s: Cell size.
    :type s: float
    :return: Bottom and top y.
    :rtype: tuple[float, float]
    """
    top = yc + len(colors) * s / 2
    for k, c in enumerate(colors):
        by = top - (k + 1) * s
        quad(ax, [(x0, by), (x0 + s, by), (x0 + s, by + s), (x0, by + s)], c)
    ax.add_patch(Rectangle((x0, top - len(colors) * s), s, len(colors) * s, facecolor="none",
                           edgecolor=INK, lw=LW_EDGE))
    return top - len(colors) * s, top


def arrow(ax, x0: float, x1: float, y: float, label: str = "", size: str = "big",
          y1: float | None = None, fontsize: float = 9, below: str = "") -> None:
    """Draw an arrow from (x0, y) to (x1, y1 or y) with optional labels above and below its middle.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Tail x.
    :type x0: float
    :param x1: Head x.
    :type x1: float
    :param y: Tail y.
    :type y: float
    :param label: Text above the arrow.
    :type label: str
    :param size: "big" between steps, "small" inside a step.
    :type size: str
    :param y1: Head y; same as the tail if None.
    :type y1: float | None
    :param fontsize: Label size.
    :type fontsize: float
    :param below: Text under the arrow.
    :type below: str
    """
    y1 = y if y1 is None else y1
    lw, scale = (1.6, 16) if size == "big" else (0.9, 9)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=lw, mutation_scale=scale,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, max(y, y1) + 0.45, label, ha="center", va="bottom",
                fontsize=fontsize, color=INK_2, linespacing=1.15)
    if below:
        ax.text((x0 + x1) / 2, min(y, y1) - 0.45, below, ha="center", va="top",
                fontsize=fontsize, color=INK_2, linespacing=1.15)


def key(ax, xc: float, y: float, cmap, labels: tuple) -> None:
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


def caption(ax, x: float, y: float, text: str) -> None:
    """Write a caption centred under a diagram.

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


# ---- CNN architecture figures ---------------------------------------------------
# One colour per operation; a box is coloured by the operation that made it.
OP_COLOR = {"1×1 conv": "#4fb37f", "3×3 conv": "#5b9bea", "max pool": "#e9706f"}
OP_LEGEND = {"1×1 conv": "1×1 conv + BN + ReLU (band mixing)",
             "3×3 conv": "3×3 conv + BN + ReLU",
             "max pool": "2×2 max pool"}
LEAF_GREEN = "#a8dcbc"
# Leaf mask drawn on the input cube's front face (row 0 at the bottom): 1 = leaf, 0 = padding.
LEAF_MASK = [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 0, 1, 0]]
CUBE_DX, CUBE_DY, CUBE_NB = 0.25, 0.18, 8
H_MAX = 4.2     # drawn height of a full-canvas feature map
W_PER_CH = 0.2  # drawn width per sqrt(channel)
DEPTH_X, DEPTH_Y = 0.30, 0.22
CHAR_W = 0.34   # rough width of one 8 pt character in drawing units, to space the legend


def backbone_stages(widths: tuple, canvas: int) -> list[tuple[str, int, int]]:
    """List the feature maps the shared backbone makes, in the order of model.forward.

    :param widths: (c0, c1, c2) channel widths from model.WIDTHS.
    :type widths: tuple
    :param canvas: Canvas side S in pixels.
    :type canvas: int
    :return: (operation, channels, map side) per feature map.
    :rtype: list[tuple[str, int, int]]
    """
    c0, c1, c2 = widths
    s = canvas
    return [("1×1 conv", c0, s), ("3×3 conv", c1, s), ("max pool", c1, s // 2),
            ("3×3 conv", c2, s // 2), ("max pool", c2, s // 4),
            ("3×3 conv", c2, s // 4), ("3×3 conv", c2, s // 4)]


def draw_backbone(ax, stages: list, canvas: int, n_bands: int, yc: float,
                  gap: float = 2.2) -> dict:
    """Draw the input cube and every backbone feature map as boxes joined by arrows.

    Box height and depth grow with sqrt(map side) and width with sqrt(channels), so
    the 16-channel full-size map does not dwarf the rest; the labels give exact sizes.
    Arrows carry no text: a box's colour names the operation that made it (legend).

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param stages: Output of :func:`backbone_stages`.
    :type stages: list
    :param canvas: Canvas side S in pixels.
    :type canvas: int
    :param n_bands: Input bands (204 for the full spectrum).
    :type n_bands: int
    :param yc: Centre line of the drawing.
    :type yc: float
    :param gap: Arrow length between boxes.
    :type gap: float
    :return: x of the last box's right edge ("x_end"), top of the tallest drawing
        ("top"), x span of the input cube ("cube") and of the boxes ("boxes").
    :rtype: dict
    """
    n = len(LEAF_MASK)
    front = [[LEAF_GREEN if m else FRONT for m in r] for r in LEAF_MASK]
    cw, ch = voxel_cube(ax, 0.0, yc - n / 2, n, n, CUBE_NB, CUBE_DX, CUBE_DY, front)
    ax.text(cw / 2, yc - n / 2 + ch + 0.25, f"{canvas} × {canvas} × {n_bands}", ha="center",
            va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    x, x_box0 = cw, None
    for op, c, s in stages:
        h = H_MAX * (s / canvas) ** 0.5
        w = W_PER_CH * c ** 0.5
        arrow(ax, x + 0.3, x + gap - 0.3, yc)
        x += gap
        x_box0 = x if x_box0 is None else x_box0
        box(ax, x, yc, w, h, h, OP_COLOR[op], DEPTH_X, DEPTH_Y)
        ax.text(x + (w + h * DEPTH_X) / 2, yc + h / 2 + h * DEPTH_Y + 0.2, f"{s}×{s}×{c}",
                ha="center", va="bottom", fontsize=7.5, color=INK_2)
        x += w + h * DEPTH_X
    return {"x_end": x, "top": yc - n / 2 + ch + 0.9, "cube": (0.0, cw), "boxes": (x_box0, x)}


def vector_cells(n: int = 8) -> list:
    """Colours for a drawn feature vector: a fixed made-up pattern of light blues.

    :param n: Number of cells drawn.
    :type n: int
    :return: One colour per cell.
    :rtype: list
    """
    levels = [0.55, 0.15, 0.75, 0.35, 0.05, 0.6, 0.25, 0.45, 0.7, 0.3]
    return [shade(OP_COLOR["3×3 conv"], levels[k % len(levels)]) for k in range(n)]


def legend(ax, x: float, y: float, items: list[tuple]) -> float:
    """Draw a one-line legend: a square swatch then its text, spaced by text length.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x: Left x.
    :type x: float
    :param y: Centre y.
    :type y: float
    :param items: (colour, text) pairs.
    :type items: list[tuple]
    :return: x where the legend ends.
    :rtype: float
    """
    bx = x
    for color, text in items:
        quad(ax, [(bx, y - 0.3), (bx + 0.6, y - 0.3), (bx + 0.6, y + 0.3), (bx, y + 0.3)],
             color, LW_CELL)
        ax.text(bx + 0.9, y, text, ha="left", va="center", fontsize=8, color=INK_2)
        bx += 0.9 + CHAR_W * len(text) + 1.4
    return bx


# ---- Disease CNN figure (both disease folders) -----------------------------------
ATTN_COLOR = "#8a5cc7"   # purple, so attention is not confused with the red max-pool boxes
# Made-up attention weights on a 4 × 4 patch of the last map (row 0 at the top);
# 0 marks positions off the leaf, which attention never weights.
EXAMPLE_ATTN = [[0.0, 0.05, 0.04, 0.0],
                [0.04, 0.10, 0.30, 0.06],
                [0.03, 0.08, 0.14, 0.04],
                [0.0, 0.04, 0.08, 0.0]]
VEC_CELLS, VEC_S = 6, 0.4   # drawn vectors are short so the two branches fit


def attention_map(ax, x0: float, yc: float, s: float = 0.5) -> None:
    """Draw the made-up attention weights as a grid of cells, darker = more weight.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x0: Left x.
    :type x0: float
    :param yc: Centre y.
    :type yc: float
    :param s: Cell size.
    :type s: float
    """
    n, w_max = len(EXAMPLE_ATTN), max(max(r) for r in EXAMPLE_ATTN)
    top = yc + n * s / 2
    for j, r in enumerate(EXAMPLE_ATTN):
        for i, w in enumerate(r):
            bx, by = x0 + i * s, top - (j + 1) * s
            face = "white" if w == 0 else shade(ATTN_COLOR, 0.9 - 0.9 * w / w_max)
            quad(ax, [(bx, by), (bx + s, by), (bx + s, by + s), (bx, by + s)], face)


def fig_disease_cnn(r: dict, path) -> None:
    """Draw the disease CNN left to right, forking into its two pooling options, and save it.

    The backbone is the variety CNN's. After it the figure splits into mean pooling and
    gated attention pooling (Ilse et al. 2018), each its own model ending in P(disease).

    :param r: canvas, n_bands, stages (from :func:`backbone_stages`), pooled (vector width),
        header (title line) and cube_note (line under the cube caption).
    :type r: dict
    :param path: PNG path.
    :type path: pathlib.Path
    """
    fig, ax = new_axes(12, 3.6)
    yc, y_cap = 3.2, -1.0
    y_up, y_dn = yc + 2.2, yc - 2.2     # the mean-pool and attention-pool branches
    b = draw_backbone(ax, r["stages"], r["canvas"], r["n_bands"], yc)
    vec = vector_cells(VEC_CELLS)

    # Fork: one line from the last map splits into the two branches (two separate models).
    xs = b["x_end"] + 1.0
    ax.plot([b["x_end"] + 0.3, xs], [yc, yc], color=INK_2, lw=1.6, solid_capstyle="butt")
    ax.plot([xs, xs], [y_dn, y_up], color=INK_2, lw=1.6, solid_capstyle="butt")

    # Attention branch: score each leaf position, softmax over the leaf → weights that
    # sum to 1, then a weighted sum of the features.
    xa = xs + 3.6
    arrow(ax, xs, xa - 0.3, y_dn, "gated\nattention", fontsize=8)
    attention_map(ax, xa, y_dn)
    ax.text(xa + 1.0, y_dn - 1.15, "weights, sum 1", ha="center", va="top", fontsize=7.5,
            color=INK_2)
    xv = xa + 2.0 + 3.8
    arrow(ax, xa + 2.3, xv - 0.3, y_dn, "weighted\nsum", fontsize=8)

    # Mean branch: every interior leaf position gets the same weight.
    arrow(ax, xs, xv - 0.3, y_up, "mean over leaf", fontsize=8)

    # Each branch: vector → dropout + linear → sigmoid → P(disease).
    xo = xv + VEC_S + 4.2
    for y in (y_up, y_dn):
        _lo, hi = column(ax, xv, y, vec, VEC_S)
        ax.text(xv + VEC_S / 2, hi + 0.15, f"{r['pooled']}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
        arrow(ax, xv + VEC_S + 0.3, xo - 0.3, y, "dropout +\nlinear", fontsize=8,
              below="sigmoid")
        column(ax, xo, y, [shade(INK, 0.45)], 0.75)
        ax.text(xo + 1.0, y, "P(disease)", ha="left", va="center", fontsize=8.5, color=INK)
    for y, name in ((y_up, "mean pool"), (y_dn, "attention pool")):
        ax.text(xo + 1.0, y - 0.55, name, ha="left", va="top", fontsize=8, color=INK_2,
                style="italic")

    ax.text(0.0, b["top"] + 0.9, r["header"], ha="left", va="bottom", fontsize=10, color=INK,
            fontweight="bold")
    c0, c1 = b["cube"]
    caption(ax, (c0 + c1) / 2, y_cap, "Leaf cube, SNV per pixel")
    ax.text((c0 + c1) / 2, y_cap - 0.95, r["cube_note"], ha="center", va="top", fontsize=8,
            color=INK_2)
    caption(ax, sum(b["boxes"]) / 2, y_cap, "Feature maps (as for variety)")
    caption(ax, xv + VEC_S / 2, y_cap, "Leaf vector")
    caption(ax, xo + 2.0, y_cap, "Prediction")
    ax.text(xo + 2.0, y_cap - 0.95, "diseased if P ≥ 0.5", ha="center", va="top", fontsize=8,
            color=INK_2)

    items = [(LEAF_GREEN, "leaf pixel (padding = 0)")] + \
        [(OP_COLOR[op], text) for op, text in OP_LEGEND.items()] + \
        [(shade(ATTN_COLOR, 0.3), "attention weight")]
    x_end = legend(ax, 0.0, -3.6, items)
    ax.text(x_end, -3.6, "BN and pooling use leaf positions only", ha="left", va="center",
            fontsize=8, color=INK_2, style="italic")

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, xo + 5.2)
    ax.set_ylim(-4.1, b["top"] + 2.0)
    save(fig, path)
