# Whole-leaf CNN on all varieties, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.556 [0.523, 0.587] | — | — |
| band z-score + mean pool | 0.560 [0.533, 0.588] | +0.005 [-0.033, +0.045] | 0.552 ± 0.001 |
| band z-score + attention | 0.562 [0.536, 0.586] | +0.007 [-0.032, +0.043] | 0.563 ± 0.021 |
| pixel SNV + mean pool | 0.589 [0.564, 0.614] | +0.034 [-0.001, +0.069] | 0.595 ± 0.029 |
| pixel SNV + attention | 0.602 [0.578, 0.626] | +0.047 [+0.012, +0.081] | 0.588 ± 0.011 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| band z-score + mean pool | 0.557 | 0.517 | 0.567 | 0.531 | 0.606 |
| band z-score + attention | 0.547 | 0.516 | 0.587 | 0.569 | 0.587 |
| pixel SNV + mean pool | 0.417 | 0.578 | 0.637 | 0.551 | 0.598 |
| pixel SNV + attention | 0.483 | 0.599 | 0.640 | 0.592 | 0.599 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.431 | 0.435 | 0.491 | 0.609 |
| band z-score + mean pool | 0.388 | 0.616 | 0.558 | 0.511 |
| band z-score + attention | 0.451 | 0.662 | 0.499 | 0.498 |
| pixel SNV + mean pool | 0.463 | 0.530 | 0.531 | 0.559 |
| pixel SNV + attention | 0.483 | 0.473 | 0.563 | 0.562 |
