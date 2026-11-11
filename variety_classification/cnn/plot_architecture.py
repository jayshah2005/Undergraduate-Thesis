"""Poster schematic of VarietyCNN: SNV leaf cube → conv feature maps → leaf mean → 4 varieties.

A diagram, not data. Layer widths come from model.py, the canvas from the saved run's
config.json and the class names from pca_common, so the labels follow the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from helpers.diagram import (INK, INK_2, LEAF_GREEN, OP_COLOR, OP_LEGEND, arrow, backbone_stages,  # noqa: E402
                     caption, column, draw_backbone, legend, new_axes, save, shade, vector_cells)
from helpers.log import get_logger  # noqa: E402
from model import N_VARIETIES, WIDTHS  # noqa: E402
from helpers.pca_common import VARIETIES  # noqa: E402

FIG = HERE / "figures"
RUN = HERE / "results" / "all"   # the full-spectrum run; RGB uses the same canvas
N_BANDS = 204
# Made-up output probabilities for the drawing, one per variety (darker = higher).
EXAMPLE_P = [0.06, 0.09, 0.79, 0.06]


def compute() -> dict:
    """Collect the sizes the figure shows.

    :return: canvas side, input bands, backbone stages, pooled width, class names.
    :rtype: dict
    """
    canvas = json.loads((RUN / "config.json").read_text())["canvas"]
    assert len(VARIETIES) == N_VARIETIES
    return {"canvas": canvas, "n_bands": N_BANDS, "stages": backbone_stages(WIDTHS, canvas),
            "pooled": WIDTHS[-1], "classes": VARIETIES}


def fig_architecture(r: dict) -> Path:
    """Draw the network left to right and save figures/architecture.png.

    :param r: Output of :func:`compute`.
    :type r: dict
    :return: Path written.
    :rtype: Path
    """
    fig, ax = new_axes(12, 3.2)
    yc, y_cap = 2.5, -1.6
    b = draw_backbone(ax, r["stages"], r["canvas"], r["n_bands"], yc)

    # Leaf pooling: average the last map over interior leaf positions → one vector.
    xv = b["x_end"] + 4.6
    arrow(ax, b["x_end"] + 0.3, xv - 0.3, yc, "mean over\nleaf", fontsize=8)
    _lo, hi = column(ax, xv, yc, vector_cells())
    ax.text(xv + 0.27, hi + 0.2, f"{r['pooled']}", ha="center", va="bottom", fontsize=7.5,
            color=INK_2)

    # Head: dropout → linear → softmax over the varieties.
    xo = xv + 0.55 + 3.6
    arrow(ax, xv + 0.85, xo - 0.3, yc, "dropout +\nlinear", fontsize=8, below="softmax")
    s = 0.75
    _lo, hi = column(ax, xo, yc, [shade(INK, 1 - 0.85 * p) for p in EXAMPLE_P], s)
    for k, name in enumerate(r["classes"]):
        ax.text(xo + s + 0.25, hi - (k + 0.5) * s, name, ha="left", va="center", fontsize=8,
                color=INK)
    ax.text(xo + s / 2, hi + 0.2, f"{len(r['classes'])}", ha="center", va="bottom",
            fontsize=7.5, color=INK_2)

    c0, c1 = b["cube"]
    caption(ax, (c0 + c1) / 2, y_cap, "Leaf cube, SNV per pixel")
    ax.text((c0 + c1) / 2, y_cap - 0.95, f"{r['n_bands']} bands, or 3 for RGB", ha="center",
            va="top", fontsize=8, color=INK_2)
    caption(ax, sum(b["boxes"]) / 2, y_cap, "Feature maps")
    caption(ax, xv + 0.27, y_cap, "Leaf vector")
    caption(ax, xo + s + 1.3, y_cap, "Variety")

    items = [(LEAF_GREEN, "leaf pixel (padding = 0)")] + \
        [(OP_COLOR[op], text) for op, text in OP_LEGEND.items()]
    x_end = legend(ax, 0.0, -4.2, items)
    ax.text(x_end, -4.2, "BN and pooling use leaf positions only", ha="left", va="center",
            fontsize=8, color=INK_2, style="italic")

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, xo + s + 4.6)
    ax.set_ylim(-4.7, b["top"] + 0.3)
    path = FIG / "architecture.png"
    save(fig, path)
    return path


def main() -> None:
    """Write figures/architecture.png."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    log.info("[saved] figures/%s", fig_architecture(compute()).name)


if __name__ == "__main__":
    main()
