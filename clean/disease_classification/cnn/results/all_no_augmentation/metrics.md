# Whole-leaf CNN on all varieties, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.556 [0.523, 0.587] | — | — |
| band z-score + mean pool | 0.575 [0.550, 0.601] | +0.020 [-0.018, +0.058] | 0.564 ± 0.008 |
| band z-score + attention | 0.558 [0.531, 0.584] | +0.002 [-0.038, +0.042] | 0.563 ± 0.015 |
| pixel SNV + mean pool | 0.617 [0.593, 0.641] | +0.061 [+0.025, +0.097] | 0.600 ± 0.006 |
| pixel SNV + attention | 0.594 [0.574, 0.614] | +0.039 [+0.004, +0.073] | 0.570 ± 0.034 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| band z-score + mean pool | 0.542 | 0.530 | 0.600 | 0.539 | 0.579 |
| band z-score + attention | 0.553 | 0.463 | 0.572 | 0.535 | 0.613 |
| pixel SNV + mean pool | 0.594 | 0.549 | 0.612 | 0.570 | 0.591 |
| pixel SNV + attention | 0.599 | 0.541 | 0.621 | 0.577 | 0.624 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.431 | 0.435 | 0.491 | 0.609 |
| band z-score + mean pool | 0.464 | 0.616 | 0.543 | 0.500 |
| band z-score + attention | 0.396 | 0.592 | 0.535 | 0.482 |
| pixel SNV + mean pool | 0.533 | 0.493 | 0.573 | 0.565 |
| pixel SNV + attention | 0.469 | 0.491 | 0.549 | 0.499 |
