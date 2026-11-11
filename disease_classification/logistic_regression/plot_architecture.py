"""Poster schematic of the disease logistic regression: leaf mean spectrum → SNV → standardise → P(disease).

A diagram, not data (drawn by diagram.fig_logistic_regression). The same model is fit once per
task (leafroll, red blotch, both, any), for one variety or a pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from helpers.diagram import fig_logistic_regression  # noqa: E402
from helpers.log import get_logger  # noqa: E402

FIG = HERE / "figures"
N_BANDS = 204


def compute() -> dict:
    """Collect the sizes and labels the figure shows.

    :return: input bands, no class names (one yes/no output) and the notes under the captions.
    :rtype: dict
    """
    return {"n_bands": N_BANDS, "classes": None, "cube_note": f"{N_BANDS} bands, or 3 for RGB",
            "fit_note": "diseased if P ≥ 0.5"}


def main() -> None:
    """Write figures/architecture.png."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    fig_logistic_regression(compute(), FIG / "architecture.png")
    log.info("[saved] figures/architecture.png")


if __name__ == "__main__":
    main()
