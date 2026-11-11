"""Shared Markdown tables, so every report reads the same way.

Each run writes two files with one section per task (disease) or per input (variety):
- final_report.md: balanced accuracy only. Each section opens with the same table: one line
  per model, its balanced accuracy, its paired difference to logistic regression and its
  score on each fold. Extra tables (per variety, single seeds) follow it.
- metrics.md: every metric (accuracy, sensitivity, specificity, precision, F1, balanced
  accuracy) with intervals, per variety and as confusion counts, built with :func:`table`.
"""
from __future__ import annotations

ROWS = ["R1", "R2", "R3", "R4", "R5"]
# Fold k tests on row Rk. Each variety has its own rows R1-R5, so in a pooled report the test set
# of fold k is several field rows (Rk of every variety); the label says which row number.
FOLDS = [f"Fold {k + 1} ({r})" for k, r in enumerate(ROWS)]
BASELINE = "logistic regression"
BANDS_TITLE = {"all": "Full spectrum (204 bands)", "rgb": "RGB (3 bands)"}  # section titles


def fold_note(varieties: list[str]) -> str:
    """One line that defines the fold columns.

    :param varieties: The varieties whose leaves the report scores.
    :type varieties: list[str]
    :return: Markdown line.
    :rtype: str
    """
    if len(varieties) == 1:
        return (f"Fold k tests on row Rk of {varieties[0]}; the next row (R1 after R5) is validation and the "
                "other 3 train.")
    return (f"Fold k tests on row Rk of every variety ({len(varieties)} field rows, since each variety has its own "
            "rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.")


def score(value: float, ci: list[float]) -> str:
    """Format a score as '0.577 [0.545, 0.610]'; an undefined bound prints as —."""
    lo, hi = (f"{c:.3f}" if c == c else "—" for c in ci)
    return f"{value:.3f} [{lo}, {hi}]"


def delta(value: float, ci: list[float]) -> str:
    """Format a signed difference as '+0.028 [−0.009, +0.061]'."""
    return f"{value:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


METRICS_INTRO = ("Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative "
                 "leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean "
                 "of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for "
                 "variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and "
                 "differences to logistic regression: `final_report.md`.")


def cell(value: float, ci: list[float] | None = None) -> str:
    """Format a value, with its interval when given; — for an undefined value (e.g. no positive calls)."""
    if value != value:  # NaN
        return "—"
    return f"{value:.3f}" if ci is None else score(value, ci)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table from already formatted cells.

    :param header: Column names.
    :type header: list[str]
    :param rows: One list of cells per line.
    :type rows: list[list[str]]
    :return: Markdown lines.
    :rtype: list[str]
    """
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(r) + " |" for r in rows])


def main_table(models: list[dict]) -> list[str]:
    """The table every section opens with.

    :param models: One dict per line: name, bal_acc, ci95, folds (5 scores) and, except for the
        baseline, delta and delta_ci95 (difference to logistic regression on the same leaves).
    :type models: list[dict]
    :return: Markdown lines.
    :rtype: list[str]
    """
    lines = ["| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | " + " | ".join(FOLDS) + " |",
             "|---|---|---|" + "---|" * len(FOLDS)]
    for m in models:
        d = delta(m["delta"], m["delta_ci95"]) if "delta" in m else "—"
        lines.append(f"| {m['name']} | {score(m['bal_acc'], m['ci95'])} | {d} | "
                     + " | ".join(f"{a:.3f}" for a in m["folds"]) + " |")
    return lines


def column_table(caption: str, columns: list[str], models: dict[str, dict]) -> list[str]:
    """A caption and a table with one line per model, e.g. balanced accuracy per variety.

    :param caption: Line above the table, e.g. "Balanced accuracy per variety:".
    :type caption: str
    :param columns: Column names, e.g. variety names.
    :type columns: list[str]
    :param models: {model name: {column: value}}; a missing column prints as —.
    :type models: dict
    :return: Markdown lines.
    :rtype: list[str]
    """
    lines = [caption, "", "| model | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, values in models.items():
        cells = [f"{values[c]:.3f}" if c in values else "—" for c in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def seed_table(models: dict[str, list[float]]) -> list[str]:
    """Balanced accuracy of each seed alone, for models that average several seeds.

    :param models: {model name: score of each seed}.
    :type models: dict
    :return: Markdown lines.
    :rtype: list[str]
    """
    lines = ["Single seeds (the ensemble above averages their probabilities):", "",
             "| model | seeds | mean ± SD |", "|---|---|---|"]
    for name, s in models.items():
        mean = sum(s) / len(s)
        sd = (sum((x - mean) ** 2 for x in s) / (len(s) - 1)) ** 0.5 if len(s) > 1 else 0.0
        lines.append(f"| {name} | {', '.join(f'{x:.3f}' for x in s)} | {mean:.3f} ± {sd:.3f} |")
    return lines
