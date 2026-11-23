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
from helpers.backbones import BACKBONES, DEFAULT_BACKBONE, WITHOUT_BLOCK_ATTENTION  # noqa: E402
from helpers.diagram import backbone_stages, fig_disease_cnn, resnet_stages  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from logistic_regression import load_meta  # noqa: E402
from model import WIDTHS  # noqa: E402
from run_within_variety import WITHIN  # noqa: E402
from train import CANVAS, canvas_for  # noqa: E402

FIG = HERE / "figures"
N_BANDS = 204


def compute() -> dict:
    """Collect the sizes and labels the figure shows.

    :return: canvas, input bands, backbone stages, pooled width, header and cube note;
        "others": {backbone: the same for each residual backbone in helpers/backbones.py}.
    :rtype: dict
    """
    counts = ", ".join(f"{v} {len(load_meta(v))}" for v in WITHIN)
    r = {"canvas": CANVAS, "n_bands": N_BANDS, "stages": backbone_stages(WIDTHS, CANVAS),
         "pooled": WIDTHS[-1], "header": f"One CNN per variety, trained on its own leaves ({counts})",
         "cube_note": f"{N_BANDS} bands, or 3 for RGB", "others": {}}
    for bb, cls in BACKBONES.items():
        # The ECA and CBAM backbones keep their plain backbone's layout, so they get no diagram of their own.
        if bb == DEFAULT_BACKBONE or bb in WITHOUT_BLOCK_ATTENTION:
            continue
        net, canvas = cls(N_BANDS), canvas_for(bb)
        r["others"][bb] = {**r, "canvas": canvas, "stages": resnet_stages(net, canvas), "pooled": net.out_channels,
                           "maps_caption": "Feature maps (as for variety)" if bb == "small_resnet" else "Feature maps"}
    return r


def main() -> None:
    """Write figures/architecture.png and, at the same scale, figures/architecture_<backbone>.png for the others."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    r = compute()
    ipu = fig_disease_cnn(r, FIG / "architecture.png", header=False)
    for bb, rb in r["others"].items():
        fig_disease_cnn({**rb, "ipu": ipu}, FIG / f"architecture_{bb}.png", header=False)
    log.info("[saved] figures/architecture*.png")


if __name__ == "__main__":
    main()
