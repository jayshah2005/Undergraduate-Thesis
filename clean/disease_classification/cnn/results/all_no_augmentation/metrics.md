# Whole-leaf CNN on all varieties: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same variety and row only (0.5 = chance). Δ: paired difference to the baseline. Bal. acc. isn't compared within a variety, so recognising a variety can raise it.

## Any disease vs healthy (772 + / 226 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.523 [0.478, 0.566] | — | 0.556 |
| band z-score + mean pool | 0.523 ± 0.015 | 0.532 [0.488, 0.577] | +0.009 [-0.045, +0.063] | 0.575 |
| band z-score + attention | 0.518 ± 0.010 | 0.518 [0.477, 0.561] | -0.005 [-0.061, +0.051] | 0.558 |
| pixel SNV + mean pool | 0.531 ± 0.042 | 0.549 [0.504, 0.594] | +0.026 [-0.025, +0.077] | 0.617 |
| pixel SNV + attention | 0.539 ± 0.016 | 0.536 [0.493, 0.581] | +0.013 [-0.032, +0.062] | 0.594 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.460 | 0.519 | 0.620 | 0.491 | 0.523 |
| band z-score + mean pool | 0.647 | 0.539 | 0.618 | 0.369 | 0.559 |
| band z-score + attention | 0.619 | 0.492 | 0.574 | 0.389 | 0.569 |
| pixel SNV + mean pool | 0.642 | 0.554 | 0.613 | 0.435 | 0.558 |
| pixel SNV + attention | 0.642 | 0.525 | 0.601 | 0.427 | 0.543 |

AUC per variety (ensemble):

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.460 | 0.378 | 0.548 | 0.574 |
| band z-score + mean pool | 0.393 | 0.599 | 0.581 | 0.580 |
| band z-score + attention | 0.383 | 0.603 | 0.579 | 0.548 |
| pixel SNV + mean pool | 0.530 | 0.592 | 0.550 | 0.557 |
| pixel SNV + attention | 0.455 | 0.568 | 0.543 | 0.591 |
