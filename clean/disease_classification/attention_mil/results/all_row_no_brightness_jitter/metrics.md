# Whole-leaf CNN on all varieties: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same variety and row only (0.5 = chance). Δ: paired difference to the baseline. Bal. acc. isn't compared within a variety, so recognising a variety can raise it.

## Any disease vs healthy (772 + / 226 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.523 [0.478, 0.566] | — | 0.556 |
| band z-score + mean pool | 0.502 ± 0.023 | 0.519 [0.475, 0.565] | -0.004 [-0.056, +0.052] | 0.560 |
| band z-score + attention | 0.535 ± 0.005 | 0.528 [0.486, 0.572] | +0.005 [-0.051, +0.058] | 0.562 |
| pixel SNV + mean pool | 0.513 ± 0.019 | 0.507 [0.463, 0.550] | -0.016 [-0.066, +0.030] | 0.589 |
| pixel SNV + attention | 0.525 ± 0.019 | 0.545 [0.502, 0.591] | +0.022 [-0.031, +0.074] | 0.602 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.460 | 0.519 | 0.620 | 0.491 | 0.523 |
| band z-score + mean pool | 0.628 | 0.492 | 0.548 | 0.381 | 0.599 |
| band z-score + attention | 0.656 | 0.567 | 0.609 | 0.372 | 0.520 |
| pixel SNV + mean pool | 0.495 | 0.523 | 0.610 | 0.410 | 0.523 |
| pixel SNV + attention | 0.603 | 0.549 | 0.601 | 0.472 | 0.540 |

AUC per variety (ensemble):

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.460 | 0.378 | 0.548 | 0.574 |
| band z-score + mean pool | 0.355 | 0.636 | 0.584 | 0.564 |
| band z-score + attention | 0.386 | 0.622 | 0.568 | 0.587 |
| pixel SNV + mean pool | 0.480 | 0.624 | 0.487 | 0.531 |
| pixel SNV + attention | 0.496 | 0.596 | 0.551 | 0.572 |
