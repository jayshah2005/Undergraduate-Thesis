# Whole-leaf CNN on all varieties, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.577 [0.545, 0.610] | — | — |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.611 ± 0.019 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.610 ± 0.002 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| baseline | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.637 [0.616, 0.659] | — | — |
| CNN, mean pool | 0.682 [0.665, 0.699] | +0.045 [+0.019, +0.069] | 0.664 ± 0.012 |
| CNN, attention pool | 0.686 [0.669, 0.704] | +0.048 [+0.023, +0.074] | 0.667 ± 0.014 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| CNN, mean pool | 0.685 | 0.663 | 0.741 | 0.675 | 0.669 |
| CNN, attention pool | 0.671 | 0.653 | 0.730 | 0.681 | 0.699 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.437 | 0.473 | 0.549 | 0.515 |
| CNN, mean pool | 0.466 | 0.576 | 0.531 | 0.548 |
| CNN, attention pool | 0.474 | 0.592 | 0.538 | 0.536 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.556 [0.523, 0.587] | — | — |
| CNN, mean pool | 0.589 [0.564, 0.614] | +0.034 [-0.001, +0.069] | 0.595 ± 0.029 |
| CNN, attention pool | 0.602 [0.578, 0.626] | +0.047 [+0.012, +0.081] | 0.588 ± 0.011 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| CNN, mean pool | 0.417 | 0.578 | 0.637 | 0.551 | 0.598 |
| CNN, attention pool | 0.483 | 0.599 | 0.640 | 0.592 | 0.599 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.431 | 0.435 | 0.491 | 0.609 |
| CNN, mean pool | 0.463 | 0.530 | 0.531 | 0.559 |
| CNN, attention pool | 0.483 | 0.473 | 0.563 | 0.562 |
