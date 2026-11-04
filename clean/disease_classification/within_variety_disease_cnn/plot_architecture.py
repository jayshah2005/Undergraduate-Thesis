"""Poster schematic of the within-variety disease CNN: one model per variety, trained on its leaves only.

A diagram, not data (drawn by diagram.fig_disease_cnn). Layer widths come from ../cnn/model.py,
the canvas from train.CANVAS and the leaf counts from the leaf table, so the labels follow the code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE.parent / "logistic_regression", HERE.parent / "cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from diagram import backbone_stages, fig_disease_cnn  # noqa: E402
from log import get_logger  # noqa: E402
from logistic_regression import load_meta  # noqa: E402
from model import WIDTHS  # noqa: E402
from run_within_variety import WITHIN  # noqa: E402
from train import CANVAS  # noqa: E402

FIG = HERE / "figures"
N_BANDS = 204


def compute() -> dict:
    """Collect the sizes and labels the figure shows.

    :return: canvas, input bands, backbone stages, pooled width, header and cube note.
    :rtype: dict
    """
    counts = ", ".join(f"{v} {len(load_meta(v))}" for v in WITHIN)
    return {"canvas": CANVAS, "n_bands": N_BANDS, "stages": backbone_stages(WIDTHS, CANVAS),
            "pooled": WIDTHS[-1], "header": f"One CNN per variety, trained on its own leaves ({counts})",
            "cube_note": f"{N_BANDS} bands, or 3 for RGB"}


def main() -> None:
    """Write figures/architecture.png."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    fig_disease_cnn(compute(), FIG / "architecture.png")
    log.info("[saved] figures/architecture.png")


if __name__ == "__main__":
    main()
