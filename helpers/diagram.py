"""Drawing pieces shared by the poster schematics (pipeline, CNN architectures).

Every schematic draws in one set of matplotlib axes where one cell is one unit wide,
with the same ink, grays, arrows and caption style, so the figures match on a poster.
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
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


def inches_per_unit(fig, ax) -> float:
    """How many inches one drawing unit takes once the axes limits are set (equal aspect).

    :param fig: Figure.
    :type fig: matplotlib.figure.Figure
    :param ax: Its axes, with limits set.
    :type ax: matplotlib.axes.Axes
    :return: Inches per unit.
    :rtype: float
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    pos = ax.get_position(original=True)   # the axes box before equal aspect shrinks one side
    w, h = fig.get_size_inches()
    return min(w * pos.width / (x1 - x0), h * pos.height / abs(y1 - y0))


def match_scale(fig, ax, ipu: float | None) -> float:
    """Resize the figure so one drawing unit takes ipu inches, as in another figure.

    Text has a fixed size in points, so two figures drawn at the same scale show boxes and
    labels in the same proportions, however wide each drawing is.

    :param fig: Figure.
    :type fig: matplotlib.figure.Figure
    :param ax: Its axes, with limits set.
    :type ax: matplotlib.axes.Axes
    :param ipu: Inches per unit to match; None leaves the figure as it is.
    :type ipu: float | None
    :return: The figure's inches per unit after resizing.
    :rtype: float
    """
    own = inches_per_unit(fig, ax)
    if ipu is None:
        return own
    fig.set_size_inches(fig.get_size_inches() * ipu / own)
    return ipu


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
OP_COLOR = {"7×7 conv": "#2f8f8a", "1×1 conv": "#4fb37f", "3×3 conv": "#5b9bea", "3×3 max pool": "#e9706f",
            "max pool": "#e9706f"}
# Legend text per operation, in legend order; a figure lists only the operations it draws.
OP_LEGEND = {"7×7 conv": "7×7 conv, stride 2 + BN + ReLU",
             "1×1 conv": "1×1 conv + BN + ReLU (band mixing)",
             "3×3 conv": "3×3 conv + BN + ReLU",
             "3×3 max pool": "3×3 max pool, stride 2",
             "max pool": "2×2 max pool"}
SKIP_LEGEND = "1×1 conv + BN (on a skip)"   # the green box on a skip, when no band-mixing conv is drawn
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
FILTER_LABEL_H = 0.9  # drawn height of a two-line 7 pt filter label
SKIP_GAP = 1.2  # arrow between the two 3×3 convs of a residual block
ADD_R = 0.3     # radius of the drawn + where a residual block adds its input back
RESIDUAL_NOTE = ("residual block (He et al. 2016): the block's input skips both 3×3 convs and is added before the "
                 "last ReLU; where the block changes the channels or the map size, a 1×1 conv + BN on the skip "
                 "matches them")


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


def resnet_stages(net, canvas: int) -> list[tuple]:
    """List the feature maps of a residual backbone (helpers/backbones.py), in the order of its forward.

    :param net: A SmallResNet or SmallResNetStrided (a 1×1 band-mixing conv, then stage1–stage3),
        or a ResNet18 (a 7×7 stride-2 conv and a 3×3 stride-2 max pool, then layer1–layer4). A
        2×2 max pool sits between stages unless the next stage's first block has stride 2.
    :type net: torch.nn.Module
    :param canvas: Canvas side S in pixels.
    :type canvas: int
    :return: (operation, channels, map side) per feature map; a residual block is
        ("block", channels, map side after it, whether its skip has a 1×1 conv, stride).
    :rtype: list[tuple]
    """
    if hasattr(net, "layer1"):
        c, s = net.conv1.out_channels, canvas // 2
        out = [("7×7 conv", c, s), ("3×3 max pool", c, s // 2)]
        s //= 2
        stages = (net.layer1, net.layer2, net.layer3, net.layer4)
    else:
        c, s = net.spec.out_channels, canvas
        out = [("1×1 conv", c, s)]
        stages = (net.stage1, net.stage2, net.stage3)
    for i, stage in enumerate(stages):
        if i and stage[0].stride == 1:
            s //= 2
            out.append(("max pool", c, s))
        for block in stage:
            s //= block.stride
            c = block.conv1.out_channels
            out.append(("block", c, s, block.down is not None, block.stride))
    return out


def op_legend(stages: list) -> list[tuple]:
    """Legend swatches for the operations a figure draws, in OP_LEGEND order.

    :param stages: Output of :func:`backbone_stages` or :func:`resnet_stages`.
    :type stages: list
    :return: (colour, text) pairs.
    :rtype: list[tuple]
    """
    ops = {"3×3 conv" if st[0] == "block" else st[0] for st in stages}
    items = [(OP_COLOR[op], text) for op, text in OP_LEGEND.items() if op in ops]
    if "1×1 conv" not in ops and any(st[0] == "block" and st[3] for st in stages):
        items.append((OP_COLOR["1×1 conv"], SKIP_LEGEND))
    return items


def residual_block(ax, x: float, yc: float, stage: tuple, c_in: int, canvas: int, gap: float,
                   y_filters: float) -> float:
    """Draw one residual basic block (He et al. 2016) after the arrow that starts at x.

    Two 3×3 conv maps, then a + where the block's input is added back; the skip line leaves
    the incoming arrow, runs over both maps and joins at the +. When the channels change it
    carries a 1×1 conv + BN.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x: Tail x of the incoming arrow, as in :func:`draw_backbone`.
    :type x: float
    :param yc: Centre line of the drawing.
    :type yc: float
    :param stage: ("block", channels, map side after the block, skip has a 1×1 conv, stride of its first conv).
    :type stage: tuple
    :param c_in: Channels coming into the block.
    :type c_in: int
    :param canvas: Canvas side S in pixels.
    :type canvas: int
    :param gap: Arrow length between steps.
    :type gap: float
    :param y_filters: Top of the filter labels.
    :type y_filters: float
    :return: x for the tail of the next arrow.
    :rtype: float
    """
    _op, c, s, proj, stride = stage
    h = H_MAX * (s / canvas) ** 0.5
    w = W_PER_CH * c ** 0.5
    span = w + h * DEPTH_X                     # drawn width of one box, depth included
    x1 = x + gap
    # A stride-2 first conv has a longer filter label ("…, stride 2"), so its block spreads out.
    x2 = x1 + span + SKIP_GAP + (1.0 if stride > 1 else 0.0)
    xa = x2 + span + 0.5 + ADD_R               # centre of the +
    top = yc + h / 2 + h * DEPTH_Y             # top of the boxes, depth included
    for xb, depth, st in ((x1, c_in, stride), (x2, c, 1)):
        box(ax, xb, yc, w, h, h, OP_COLOR["3×3 conv"], DEPTH_X, DEPTH_Y)
        ax.text(xb + span / 2, y_filters, filter_label("3×3 conv", c, depth, st), ha="center", va="top",
                fontsize=7, color=INK_2, linespacing=1.2)
    # Both maps have the same size: one label sits on them, under the skip line, so it does not
    # read as an operation on the skip.
    ax.text((x1 + x2 + span) / 2, top + 0.2, f"{s}×{s}×{c}", ha="center", va="bottom", fontsize=7.5,
            color=INK_2)
    arrow(ax, x1 + span + 0.2, x2 - 0.2, yc, size="small")
    arrow(ax, x2 + span + 0.15, xa - ADD_R - 0.05, yc, size="small")
    ax.add_patch(Circle((xa, yc), ADD_R, facecolor="white", edgecolor=INK, lw=LW_EDGE * 0.8))
    ax.text(xa, yc, "+", ha="center", va="center", fontsize=9, color=INK)

    # The skip leaves the incoming arrow and runs above the size label; with a 1×1 conv on it, it runs
    # higher, through a small green box: the projected map, with its filters written above it.
    xs, ya = x + 1.2, top + (1.7 if proj else 1.2)
    if proj:
        bw, bh, bd = W_PER_CH * c ** 0.5 * 0.5, 0.8, 0.8
        bx = (xs + xa) / 2 - (bw + bd * DEPTH_X) / 2
        ax.plot([xs, xs, bx], [yc, ya, ya], color=INK_2, lw=1.1, solid_joinstyle="round")
        ax.plot([bx + bw + bd * DEPTH_X, xa], [ya, ya], color=INK_2, lw=1.1)
        box(ax, bx, ya, bw, bh, bd, OP_COLOR["1×1 conv"], DEPTH_X, DEPTH_Y)
        label = f"{c} filters, 1×1×{c_in}" + (f", stride {stride}" if stride > 1 else "") + " + BN"
        ax.text(bx + (bw + bd * DEPTH_X) / 2, ya + bh / 2 + bd * DEPTH_Y + 0.15, label, ha="center", va="bottom",
                fontsize=7, color=INK_2)
    else:
        ax.plot([xs, xs, xa], [yc, ya, ya], color=INK_2, lw=1.1, solid_joinstyle="round")
    arrow(ax, xa, xa, ya, size="small", y1=yc + ADD_R + 0.03)
    return xa + ADD_R - 0.25


def residual_note(ax, x: float, y: float) -> None:
    """Draw the legend line that explains the + and the skip of a residual block.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param x: Left x.
    :type x: float
    :param y: Centre y.
    :type y: float
    """
    ax.add_patch(Circle((x + 0.3, y), ADD_R, facecolor="white", edgecolor=INK, lw=LW_CELL))
    ax.text(x + 0.3, y, "+", ha="center", va="center", fontsize=8, color=INK)
    ax.text(x + 0.9, y, RESIDUAL_NOTE, ha="left", va="center", fontsize=8, color=INK_2)


def draw_backbone(ax, stages: list, canvas: int, n_bands: int, yc: float,
                  gap: float = 2.2) -> dict:
    """Draw the input cube and every backbone feature map as boxes joined by arrows.

    Box height and depth grow with sqrt(map side) and width with sqrt(channels), so
    the 16-channel full-size map does not dwarf the rest; the labels give exact sizes.
    Arrows carry no text: a box's colour names the operation that made it (legend).
    Under each box: the number of filters and each filter's k × k × depth shape.
    A residual block (from :func:`resnet_stages`) is drawn by :func:`residual_block`.

    :param ax: Axes to draw on.
    :type ax: matplotlib.axes.Axes
    :param stages: Output of :func:`backbone_stages` or :func:`resnet_stages`.
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
        ("top"), x span of the input cube ("cube") and of the boxes ("boxes"), and the
        lowest y the filter labels reach ("bottom").
    :rtype: dict
    """
    n = len(LEAF_MASK)
    front = [[LEAF_GREEN if m else FRONT for m in r] for r in LEAF_MASK]
    cw, ch = voxel_cube(ax, 0.0, yc - n / 2, n, n, CUBE_NB, CUBE_DX, CUBE_DY, front)
    ax.text(cw / 2, yc - n / 2 + ch + 0.25, f"{canvas} × {canvas} × {n_bands}", ha="center",
            va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    x, x_box0 = cw, None
    c_in = n_bands   # channels coming into the next layer = how deep its filters are
    y_filters = yc - n / 2 - 0.2   # one line for every filter label, level with the cube's base
    for stage in stages:
        op, c, s = stage[:3]
        h = H_MAX * (s / canvas) ** 0.5
        w = W_PER_CH * c ** 0.5
        arrow(ax, x + 0.3, x + gap - 0.3, yc)
        if op == "block":
            x_box0 = x + gap if x_box0 is None else x_box0
            x = residual_block(ax, x, yc, stage, c_in, canvas, gap, y_filters)
            c_in = c
            continue
        x += gap
        x_box0 = x if x_box0 is None else x_box0
        box(ax, x, yc, w, h, h, OP_COLOR[op], DEPTH_X, DEPTH_Y)
        xm = x + (w + h * DEPTH_X) / 2
        ax.text(xm, yc + h / 2 + h * DEPTH_Y + 0.2, f"{s}×{s}×{c}",
                ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.text(xm, y_filters, filter_label(op, c, c_in), ha="center", va="top", fontsize=7,
                color=INK_2, linespacing=1.2)
        c_in = c
        x += w + h * DEPTH_X
    return {"x_end": x, "top": yc - n / 2 + ch + 0.9, "cube": (0.0, cw), "boxes": (x_box0, x),
            "bottom": y_filters - FILTER_LABEL_H}


def filter_label(op: str, c_out: int, c_in: int, stride: int = 1) -> str:
    """Text under a feature map: how many filters made it and the shape of each one.

    A k×k conv filter spans every incoming channel, so it is k × k × c_in; its depth is the
    number of channels of the map before it (204 bands for the first, band-mixing conv).
    Max pooling has no filters; it works on each channel alone.

    :param op: Operation name, a key of OP_COLOR.
    :type op: str
    :param c_out: Channels made (= number of filters).
    :type c_out: int
    :param c_in: Channels coming in (= filter depth).
    :type c_in: int
    :param stride: The conv's stride; 2 is written after the filter shape (the 7×7 conv always has it).
    :type stride: int
    :return: Two-line label.
    :rtype: str
    """
    if op == "max pool":
        return "2×2 window\nper channel"
    if op == "3×3 max pool":
        return "3×3 window\nstride 2"
    k = op.split("×")[0]
    stride = 2 if op == "7×7 conv" else stride
    return f"{c_out} filters\n{k}×{k}×{c_in}" + (f", stride {stride}" if stride > 1 else "")


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


def fig_disease_cnn(r: dict, path, header: bool = True) -> float:
    """Draw the disease CNN left to right, forking into its two pooling options, and save it.

    The backbone is the variety CNN's. After it the figure splits into mean pooling and
    gated attention pooling (Ilse et al. 2018), each its own model ending in P(disease).

    :param r: canvas, n_bands, stages (from :func:`backbone_stages` or :func:`resnet_stages`),
        pooled (vector width), header (title line), cube_note (line under the cube caption) and,
        optionally, ipu: inches per drawing unit of another figure to match (see :func:`match_scale`).
    :type r: dict
    :param path: PNG path.
    :type path: pathlib.Path
    :param header: Draw the title line.
    :type header: bool
    :return: Inches per drawing unit, to draw another backbone at the same scale.
    :rtype: float
    """
    fig, ax = new_axes(12, 3.7)  # the same size either way, so the drawing keeps its scale
    yc, y_cap = 3.2, -1.1
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
    xv = xa + 2.0 + 3.8 + (0.5 if r["pooled"] >= 100 else 0.0)   # room for a 3-digit vector width
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

    if header:
        ax.text(0.0, b["top"] + 0.9, r["header"], ha="left", va="bottom", fontsize=10, color=INK,
                fontweight="bold")

    c0, c1 = b["cube"]
    caption(ax, (c0 + c1) / 2, y_cap, "Leaf cube, SNV per pixel")
    ax.text((c0 + c1) / 2, y_cap - 0.95, r["cube_note"], ha="center", va="top", fontsize=8,
            color=INK_2)
    caption(ax, sum(b["boxes"]) / 2, y_cap, r.get("maps_caption", "Feature maps (as for variety)"))
    caption(ax, xv + VEC_S / 2, y_cap, "Leaf vector")
    caption(ax, xo + 2.0, y_cap, "Prediction")
    ax.text(xo + 2.0, y_cap - 0.95, "diseased if P ≥ 0.5", ha="center", va="top", fontsize=8,
            color=INK_2)

    items = [(LEAF_GREEN, "leaf pixel (padding = 0)")] + op_legend(r["stages"]) + \
        [(shade(ATTN_COLOR, 0.3), "attention weight")]
    x_end = legend(ax, 0.0, -3.7, items)
    ax.text(x_end, -3.7, "BN and pooling use leaf positions only", ha="left", va="center",
            fontsize=8, color=INK_2, style="italic")
    y_low = -4.2
    if any(st[0] == "block" for st in r["stages"]):
        residual_note(ax, 0.0, -4.6)
        y_low = -5.1

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, xo + 5.2)
    # The saved image is cropped to the axes and any text past them, so without a header the
    # limit drops to the top of the drawing and no empty band is left where the header was.
    ax.set_ylim(y_low, b["top"] + (2.0 if header else 0.0))
    ipu = match_scale(fig, ax, r.get("ipu"))
    save(fig, path)
    return ipu


# ---- Logistic regression figure (variety and disease) ----------------------------
# A made-up leaf spectrum, one value per drawn cell, short wavelengths first: low in the blue,
# a green bump, low in the red, then the high near-infrared plateau of a healthy leaf.
EXAMPLE_SPECTRUM = [0.05, 0.08, 0.16, 0.09, 0.05, 0.28, 0.46, 0.50, 0.50, 0.48]
# Made-up values after per-band standardisation (each band centred on the training leaves).
EXAMPLE_STANDARDISED = [-0.4, 0.9, -1.2, 0.3, 1.5, -0.8, 0.2, -1.6, 0.7, 1.1]
SPEC_S = 0.45   # cell size of the drawn spectra
LR_STEP = 5.6   # distance from one spectrum to the next, wide enough for the captions


def fig_logistic_regression(r: dict, path) -> None:
    """Draw the logistic-regression pipeline left to right, in the CNN figures' style, and save it.

    Leaf cube → mean over leaf pixels → SNV → per-band standardisation → linear → softmax
    (several classes) or sigmoid (one disease). The weights have the depth of the spectrum,
    as a CNN filter has the depth of the map it reads, and the label under the linear arrow
    says so.

    :param r: n_bands, classes (names, or None for one yes/no disease), example_p (one made-up
        probability per class, ignored when classes is None), cube_note (line under the cube
        caption), fit_note (line under the output caption).
    :type r: dict
    :param path: PNG path.
    :type path: pathlib.Path
    """
    fig, ax = new_axes(8.6, 3.2)
    yc = 2.5
    n = len(LEAF_MASK)
    y_cap = yc - n / 2 - 0.6

    front = [[LEAF_GREEN if m else FRONT for m in row_] for row_ in LEAF_MASK]
    cw, ch = voxel_cube(ax, 0.0, yc - n / 2, n, n, CUBE_NB, CUBE_DX, CUBE_DY, front)
    ax.text(cw / 2, yc - n / 2 + ch + 0.25, f"leaf pixels × {r['n_bands']}", ha="center",
            va="bottom", fontsize=8.5, color=INK, fontweight="bold")

    # The three spectra: reflectance, SNV (centred and scaled over its own bands), standardised.
    spec = np.array(EXAMPLE_SPECTRUM)
    snv_ = (spec - spec.mean()) / spec.std()
    spectra = [[REFLECT(float(np.clip(v / 0.55, 0, 1))) for v in spec],
               [SNV_MAP(float(np.clip((v + 2) / 4, 0, 1))) for v in snv_],
               [SNV_MAP(float(np.clip((v + 2) / 4, 0, 1))) for v in EXAMPLE_STANDARDISED]]
    steps = [("mean over\nleaf pixels", "Mean spectrum", "one per leaf"),
             ("SNV", "SNV", "over the leaf's\nown bands"),
             ("standardise\nper band", "Standardised", "mean, sd from\ntraining leaves")]
    x_prev, x = cw, cw + 4.5
    for cells, (verb, name, note) in zip(spectra, steps):
        arrow(ax, x_prev + 0.3, x - 0.3, yc, verb, fontsize=8)
        _lo, hi = column(ax, x, yc, cells, SPEC_S)
        ax.text(x + SPEC_S / 2, hi + 0.15, f"{r['n_bands']}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
        caption(ax, x + SPEC_S / 2, y_cap, name)
        ax.text(x + SPEC_S / 2, y_cap - 0.95, note, ha="center", va="top", fontsize=8, color=INK_2,
                linespacing=1.15)
        x_prev, x = x + SPEC_S, x + SPEC_S + LR_STEP - SPEC_S

    # Linear layer: one weight per band for each class (one class for a yes/no disease).
    classes = r["classes"]
    k = len(classes) if classes else 1
    xo = x + 0.6
    arrow(ax, x_prev + 0.3, xo - 0.3, yc, "linear", fontsize=8,
          below="softmax" if classes else "sigmoid")
    weights = f"{k} × {r['n_bands']} weights\n+ {k} biases" if k > 1 else \
        f"{r['n_bands']} weights\n+ 1 bias"
    ax.text((x_prev + xo) / 2, yc - 1.25, weights, ha="center", va="top", fontsize=7,
            color=INK_2, linespacing=1.2)

    s = 0.75
    if classes:
        _lo, hi = column(ax, xo, yc, [shade(INK, 1 - 0.85 * p) for p in r["example_p"]], s)
        for j, name in enumerate(classes):
            ax.text(xo + s + 0.25, hi - (j + 0.5) * s, name, ha="left", va="center",
                    fontsize=8, color=INK)
        ax.text(xo + s / 2, hi + 0.2, f"{k}", ha="center", va="bottom", fontsize=7.5,
                color=INK_2)
        out_name, x_out = "Variety", xo + s + 1.3
    else:
        column(ax, xo, yc, [shade(INK, 0.45)], s)
        ax.text(xo + s + 0.25, yc, "P(disease)", ha="left", va="center", fontsize=8.5, color=INK)
        out_name, x_out = "Prediction", xo + 1.8
    caption(ax, (0 + cw) / 2, y_cap, "Leaf cube")
    ax.text(cw / 2, y_cap - 0.95, r["cube_note"], ha="center", va="top", fontsize=8, color=INK_2)
    caption(ax, x_out, y_cap, out_name)
    ax.text(x_out, y_cap - 0.95, r["fit_note"], ha="center", va="top", fontsize=8, color=INK_2,
            linespacing=1.15)

    # Legend: leaf pixels, then the two colour scales the spectra use.
    y_leg = y_cap - 3.0
    bx = legend(ax, 0.0, y_leg, [(LEAF_GREEN, "leaf pixel")])
    for title, cmap, labels in (("reflectance", REFLECT, ("low", "", "high")),
                                ("SNV, standardised", SNV_MAP, ("below", "0", "above"))):
        ax.text(bx, y_leg, title, ha="left", va="center", fontsize=8, color=INK_2)
        bx += CHAR_W * len(title) + 0.6
        key(ax, bx + 2.0, y_leg - 0.17, cmap, labels)
        bx += 4.0 + 1.4

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, max(xo + s + 5.0, bx))
    ax.set_ylim(y_leg - 0.9, yc - n / 2 + ch + 1.0)
    save(fig, path)
