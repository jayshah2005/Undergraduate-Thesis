# Whole-leaf CNN on all varieties, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.584 [0.558, 0.612] | — | — |
| band z-score + mean pool | 0.595 [0.571, 0.619] | +0.011 [-0.021, +0.042] | 0.574 ± 0.023 |
| band z-score + attention | 0.592 [0.567, 0.616] | +0.008 [-0.027, +0.042] | 0.591 ± 0.026 |
| pixel SNV + mean pool | 0.614 [0.592, 0.635] | +0.030 [+0.001, +0.056] | 0.611 ± 0.006 |
| pixel SNV + attention | 0.597 [0.575, 0.619] | +0.013 [-0.016, +0.041] | 0.600 ± 0.002 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| band z-score + mean pool | 0.543 | 0.567 | 0.650 | 0.577 | 0.639 |
| band z-score + attention | 0.550 | 0.540 | 0.649 | 0.565 | 0.623 |
| pixel SNV + mean pool | 0.527 | 0.567 | 0.606 | 0.645 | 0.651 |
| pixel SNV + attention | 0.491 | 0.543 | 0.606 | 0.627 | 0.618 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.449 | 0.495 | 0.542 | 0.541 |
| band z-score + mean pool | 0.432 | 0.483 | 0.504 | 0.564 |
| band z-score + attention | 0.446 | 0.461 | 0.524 | 0.515 |
| pixel SNV + mean pool | 0.487 | 0.563 | 0.500 | 0.536 |
| pixel SNV + attention | 0.521 | 0.538 | 0.500 | 0.493 |
