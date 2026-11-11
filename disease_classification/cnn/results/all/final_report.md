# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.611, 0.630, 0.591 | 0.611 ± 0.019 |
| CNN, attention pool | 0.612, 0.607, 0.611 | 0.610 ± 0.002 |

Median best epoch: CNN, mean pool 20, CNN, attention pool 16.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | — | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| CNN, mean pool | 0.682 [0.665, 0.699] | +0.045 [+0.019, +0.069] | 0.685 | 0.663 | 0.741 | 0.675 | 0.669 |
| CNN, attention pool | 0.686 [0.669, 0.704] | +0.048 [+0.023, +0.074] | 0.671 | 0.653 | 0.730 | 0.681 | 0.699 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.437 | 0.473 | 0.549 | 0.515 |
| CNN, mean pool | 0.466 | 0.576 | 0.531 | 0.548 |
| CNN, attention pool | 0.474 | 0.592 | 0.538 | 0.536 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.653, 0.662, 0.677 | 0.664 ± 0.012 |
| CNN, attention pool | 0.682, 0.656, 0.663 | 0.667 ± 0.014 |

Median best epoch: CNN, mean pool 9, CNN, attention pool 10.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | — | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| CNN, mean pool | 0.589 [0.564, 0.614] | +0.034 [-0.001, +0.069] | 0.417 | 0.578 | 0.637 | 0.551 | 0.598 |
| CNN, attention pool | 0.602 [0.578, 0.626] | +0.047 [+0.012, +0.081] | 0.483 | 0.599 | 0.640 | 0.592 | 0.599 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.431 | 0.435 | 0.491 | 0.609 |
| CNN, mean pool | 0.463 | 0.530 | 0.531 | 0.559 |
| CNN, attention pool | 0.483 | 0.473 | 0.563 | 0.562 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.565, 0.597, 0.622 | 0.595 ± 0.029 |
| CNN, attention pool | 0.577, 0.589, 0.598 | 0.588 ± 0.011 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 7.
