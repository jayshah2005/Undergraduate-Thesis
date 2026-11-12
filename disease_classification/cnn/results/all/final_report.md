# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | +0.014 [-0.022, +0.048] | 0.552 | 0.623 | 0.550 | 0.518 | 0.676 |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | +0.045 [+0.006, +0.082] | 0.587 | 0.703 | 0.616 | 0.541 | 0.613 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |
| small ResNet, mean pool | 0.510 | 0.445 | 0.509 |
| small ResNet, attention pool | 0.507 | 0.489 | 0.513 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.611, 0.630, 0.591 | 0.611 ± 0.019 |
| CNN, attention pool | 0.612, 0.607, 0.611 | 0.610 ± 0.002 |
| small ResNet, mean pool | 0.597, 0.599, 0.557 | 0.584 ± 0.024 |
| small ResNet, attention pool | 0.578, 0.598, 0.605 | 0.593 ± 0.014 |

Median best epoch: CNN, mean pool 20, CNN, attention pool 16, small ResNet, mean pool 16, small ResNet, attention pool 18.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | — | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| CNN, mean pool | 0.682 [0.665, 0.699] | +0.045 [+0.019, +0.069] | 0.685 | 0.663 | 0.741 | 0.675 | 0.669 |
| CNN, attention pool | 0.686 [0.669, 0.704] | +0.048 [+0.023, +0.074] | 0.671 | 0.653 | 0.730 | 0.681 | 0.699 |
| small ResNet, mean pool | 0.675 [0.657, 0.694] | +0.038 [+0.011, +0.065] | 0.620 | 0.651 | 0.740 | 0.677 | 0.676 |
| small ResNet, attention pool | 0.686 [0.669, 0.703] | +0.048 [+0.024, +0.074] | 0.662 | 0.652 | 0.752 | 0.677 | 0.696 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.437 | 0.473 | 0.549 | 0.515 |
| CNN, mean pool | 0.466 | 0.576 | 0.531 | 0.548 |
| CNN, attention pool | 0.474 | 0.592 | 0.538 | 0.536 |
| small ResNet, mean pool | 0.500 | 0.518 | 0.554 | 0.529 |
| small ResNet, attention pool | 0.503 | 0.582 | 0.531 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.653, 0.662, 0.677 | 0.664 ± 0.012 |
| CNN, attention pool | 0.682, 0.656, 0.663 | 0.667 ± 0.014 |
| small ResNet, mean pool | 0.661, 0.647, 0.670 | 0.659 ± 0.012 |
| small ResNet, attention pool | 0.671, 0.682, 0.627 | 0.660 ± 0.029 |

Median best epoch: CNN, mean pool 9, CNN, attention pool 10, small ResNet, mean pool 11, small ResNet, attention pool 13.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | — | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| CNN, mean pool | 0.589 [0.564, 0.614] | +0.034 [-0.001, +0.069] | 0.417 | 0.578 | 0.637 | 0.551 | 0.598 |
| CNN, attention pool | 0.602 [0.578, 0.626] | +0.047 [+0.012, +0.081] | 0.483 | 0.599 | 0.640 | 0.592 | 0.599 |
| small ResNet, mean pool | 0.596 [0.571, 0.621] | +0.041 [+0.007, +0.077] | 0.600 | 0.584 | 0.613 | 0.573 | 0.613 |
| small ResNet, attention pool | 0.613 [0.591, 0.635] | +0.057 [+0.025, +0.093] | 0.497 | 0.553 | 0.588 | 0.603 | 0.611 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.431 | 0.435 | 0.491 | 0.609 |
| CNN, mean pool | 0.463 | 0.530 | 0.531 | 0.559 |
| CNN, attention pool | 0.483 | 0.473 | 0.563 | 0.562 |
| small ResNet, mean pool | 0.481 | 0.532 | 0.530 | 0.507 |
| small ResNet, attention pool | 0.602 | 0.666 | 0.550 | 0.495 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.565, 0.597, 0.622 | 0.595 ± 0.029 |
| CNN, attention pool | 0.577, 0.589, 0.598 | 0.588 ± 0.011 |
| small ResNet, mean pool | 0.599, 0.574, 0.570 | 0.581 ± 0.016 |
| small ResNet, attention pool | 0.606, 0.599, 0.605 | 0.604 ± 0.004 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 7, small ResNet, mean pool 17, small ResNet, attention pool 6.
