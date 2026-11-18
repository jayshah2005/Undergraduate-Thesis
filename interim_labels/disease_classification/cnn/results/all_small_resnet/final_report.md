# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | +0.014 [-0.022, +0.048] | 0.552 | 0.623 | 0.550 | 0.518 | 0.676 |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | +0.045 [+0.006, +0.082] | 0.587 | 0.703 | 0.616 | 0.541 | 0.613 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| small ResNet, mean pool | 0.510 | 0.445 | 0.509 |
| small ResNet, attention pool | 0.507 | 0.489 | 0.513 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.597, 0.599, 0.557 | 0.584 ± 0.024 |
| small ResNet, attention pool | 0.578, 0.598, 0.605 | 0.593 ± 0.014 |

Median best epoch: small ResNet, mean pool 16, small ResNet, attention pool 18.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | — | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| small ResNet, mean pool | 0.675 [0.657, 0.694] | +0.038 [+0.011, +0.065] | 0.620 | 0.651 | 0.740 | 0.677 | 0.676 |
| small ResNet, attention pool | 0.686 [0.669, 0.703] | +0.048 [+0.024, +0.074] | 0.662 | 0.652 | 0.752 | 0.677 | 0.696 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.437 | 0.473 | 0.549 | 0.515 |
| small ResNet, mean pool | 0.500 | 0.518 | 0.554 | 0.529 |
| small ResNet, attention pool | 0.503 | 0.582 | 0.531 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.661, 0.647, 0.670 | 0.659 ± 0.012 |
| small ResNet, attention pool | 0.671, 0.682, 0.627 | 0.660 ± 0.029 |

Median best epoch: small ResNet, mean pool 11, small ResNet, attention pool 13.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | — | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| small ResNet, mean pool | 0.596 [0.571, 0.621] | +0.041 [+0.007, +0.077] | 0.600 | 0.584 | 0.613 | 0.573 | 0.613 |
| small ResNet, attention pool | 0.613 [0.591, 0.635] | +0.057 [+0.025, +0.093] | 0.497 | 0.553 | 0.588 | 0.603 | 0.611 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.431 | 0.435 | 0.491 | 0.609 |
| small ResNet, mean pool | 0.481 | 0.532 | 0.530 | 0.507 |
| small ResNet, attention pool | 0.602 | 0.666 | 0.550 | 0.495 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.599, 0.574, 0.570 | 0.581 ± 0.016 |
| small ResNet, attention pool | 0.606, 0.599, 0.605 | 0.604 ± 0.004 |

Median best epoch: small ResNet, mean pool 17, small ResNet, attention pool 6.
