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
ROOT = HERE.parents[1]
for p in (ROOT, HERE.parent / "logistic_regression", HERE.parent / "cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.backbones import BACKBONES  # noqa: E402
from helpers.diagram import backbone_stages, fig_disease_cnn, resnet_stages  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from logistic_regression import load_meta  # noqa: E402
from model import SECOND_BACKBONE, WIDTHS  # noqa: E402
from run_within_variety import WITHIN  # noqa: E402
from train import CANVAS  # noqa: E402

FIG = HERE / "figures"
N_BANDS = 204


def compute() -> dict:
    """Collect the sizes and labels the figure shows.

    :return: canvas, input bands, backbone stages, pooled width, header and cube note;
        "second": the same for SECOND_BACKBONE (a SmallResNet), or None.
    :rtype: dict
    """
    counts = ", ".join(f"{v} {len(load_meta(v))}" for v in WITHIN)
    r = {"canvas": CANVAS, "n_bands": N_BANDS, "stages": backbone_stages(WIDTHS, CANVAS),
         "pooled": WIDTHS[-1], "header": f"One CNN per variety, trained on its own leaves ({counts})",
         "cube_note": f"{N_BANDS} bands, or 3 for RGB", "second": None}
    if SECOND_BACKBONE:
        net = BACKBONES[SECOND_BACKBONE](N_BANDS)
        r["second"] = {**r, "stages": resnet_stages(net, CANVAS), "pooled": net.out_channels}
    return r


def main() -> None:
    """Write figures/architecture.png and, at the same scale, figures/architecture_<SECOND_BACKBONE>.png."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    r = compute()
    ipu = fig_disease_cnn(r, FIG / "architecture.png", header=False)
    if r["second"]:
        fig_disease_cnn({**r["second"], "ipu": ipu}, FIG / f"architecture_{SECOND_BACKBONE}.png", header=False)
    log.info("[saved] figures/architecture*.png")


if __name__ == "__main__":
    main()
