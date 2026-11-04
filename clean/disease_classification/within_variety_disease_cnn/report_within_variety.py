"""Score each variety's CNNs against its mean-spectrum baseline and draw the figures (../cnn's report, per variety)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parents[1]
for p in (CLEAN_ROOT, CLEAN_ROOT / "spectral_analysis" / "PCA", HERE.parent / "logistic_regression", HERE.parent / "cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from report import write_report  # noqa: E402
from log import get_logger  # noqa: E402
from logistic_regression import BANDS  # noqa: E402
from pca_common import get_device  # noqa: E402
from run_experiments import results_root  # noqa: E402
from run_within_variety import RESULTS, WITHIN  # noqa: E402

FIG = HERE / "figures"


def main() -> None:
    """Write results/<variety>/metrics.md and figures/<variety>/ for every variety with runs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", nargs="+", choices=WITHIN, default=list(WITHIN), help="varieties to report")
    ap.add_argument("--bands", choices=BANDS, default="all", help="report the runs made with --bands <value>")
    args = ap.parse_args()
    log = get_logger("within_cnn_report")
    device = get_device()
    for variety in args.variety:
        root = results_root(False, variety=variety, bands=args.bands, results=RESULTS)
        write_report(variety, args.bands, root, FIG / root.name, device)
    log.info("[saved] results/{%s}/metrics.md", ",".join(results_root(False, variety=v, bands=args.bands,
                                                                       results=RESULTS).name for v in args.variety))


if __name__ == "__main__":
    main()
