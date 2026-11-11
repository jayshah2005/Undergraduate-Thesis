"""Poster schematic of the variety logistic regression: leaf mean spectrum → SNV → standardise → 4 varieties.

A diagram, not data (drawn by diagram.fig_logistic_regression). Class names come from
pca_common, so the labels follow the code.
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
from helpers.pca_common import VARIETIES  # noqa: E402

FIG = HERE / "figures"
N_BANDS = 204
# Made-up output probabilities for the drawing, one per variety (darker = higher).
EXAMPLE_P = [0.06, 0.09, 0.79, 0.06]


def compute() -> dict:
    """Collect the sizes and labels the figure shows.

    :return: input bands, class names, example probabilities and the notes under the captions.
    :rtype: dict
    """
    assert len(EXAMPLE_P) == len(VARIETIES)
    return {"n_bands": N_BANDS, "classes": list(VARIETIES), "example_p": EXAMPLE_P,
            "cube_note": f"{N_BANDS} bands, or 3 for RGB",
            "fit_note": "C chosen on the\nvalidation row"}


def main() -> None:
    """Write figures/architecture.png."""
    FIG.mkdir(exist_ok=True)
    log = get_logger("plot_architecture")
    fig_logistic_regression(compute(), FIG / "architecture.png")
    log.info("[saved] figures/architecture.png")


if __name__ == "__main__":
    main()
