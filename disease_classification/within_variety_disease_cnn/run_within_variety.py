"""Train one whole-leaf CNN per variety (Baco, Chardonnay, Vidal), each on that variety's leaves only.

The network, training and scoring are ../cnn's; this script only loops over the varieties and
keeps their results, logs and figures in this folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE.parent / "logistic_regression", HERE.parent / "cnn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Imported first: it sets the cuBLAS setting that must come before CUDA starts.
from run_experiments import parse_args, results_root, run, run_settings  # noqa: E402
from helpers.log import get_logger  # noqa: E402

# Cab Franc sits out: 1 leaf has leafroll and only 9 lack red blotch, too few to train on.
WITHIN = ["Baco", "Chardonnay", "Vidal"]
RESULTS, LOGS = HERE / "results", HERE / "logs"


def main() -> None:
    """Train every chosen variety (default: all three) one after the other, logging each model's balanced accuracy."""
    args = parse_args(WITHIN, list(WITHIN), "varieties to train, each on its own; results go to results/<variety>/")
    cfg, seeds, folds = run_settings(args)
    for variety in args.variety:
        root = results_root(args.smoke, args.dropout, variety, not args.no_augmentation, args.bands, RESULTS)
        # One logger per variety, so each log file gets only its own variety's lines.
        log = get_logger(f"within_cnn_{root.name}", log_file=None if args.smoke else LOGS / f"{root.name}.log")
        run(variety, args.bands, args.task, root, log, cfg, seeds, folds)
        log.info("[done] %s. For intervals, tables and figures: report_within_variety.py%s", root.relative_to(HERE),
                 " --bands rgb" if args.bands == "rgb" else "")


if __name__ == "__main__":
    main()
