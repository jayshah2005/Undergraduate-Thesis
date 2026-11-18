# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | +0.021 [-0.016, +0.055] | 0.572 | 0.662 | 0.495 | 0.513 | 0.688 |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | +0.030 [-0.009, +0.065] | 0.504 | 0.684 | 0.571 | 0.578 | 0.625 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| small ResNet (no max pool), mean pool | 0.528 | 0.458 | 0.498 |
| small ResNet (no max pool), attention pool | 0.538 | 0.467 | 0.544 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.597, 0.556, 0.589 | 0.581 ± 0.021 |
| small ResNet (no max pool), attention pool | 0.577, 0.573, 0.594 | 0.582 ± 0.011 |

Median best epoch: small ResNet (no max pool), mean pool 12, small ResNet (no max pool), attention pool 15.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | — | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| small ResNet (no max pool), mean pool | 0.673 [0.652, 0.692] | +0.035 [+0.009, +0.062] | 0.601 | 0.634 | 0.723 | 0.696 | 0.695 |
| small ResNet (no max pool), attention pool | 0.681 [0.662, 0.700] | +0.044 [+0.018, +0.070] | 0.646 | 0.644 | 0.739 | 0.673 | 0.705 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.437 | 0.473 | 0.549 | 0.515 |
| small ResNet (no max pool), mean pool | 0.485 | 0.518 | 0.533 | 0.543 |
| small ResNet (no max pool), attention pool | 0.477 | 0.520 | 0.538 | 0.543 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.670, 0.662, 0.659 | 0.663 ± 0.006 |
| small ResNet (no max pool), attention pool | 0.686, 0.639, 0.670 | 0.665 ± 0.024 |

Median best epoch: small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 8.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | — | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| small ResNet (no max pool), mean pool | 0.598 [0.574, 0.621] | +0.043 [+0.008, +0.078] | 0.583 | 0.523 | 0.646 | 0.574 | 0.556 |
| small ResNet (no max pool), attention pool | 0.580 [0.553, 0.607] | +0.025 [-0.012, +0.062] | 0.605 | 0.519 | 0.656 | 0.561 | 0.595 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.431 | 0.435 | 0.491 | 0.609 |
| small ResNet (no max pool), mean pool | 0.553 | 0.639 | 0.523 | 0.510 |
| small ResNet (no max pool), attention pool | 0.407 | 0.607 | 0.568 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.614, 0.579, 0.558 | 0.583 ± 0.028 |
| small ResNet (no max pool), attention pool | 0.583, 0.548, 0.533 | 0.555 ± 0.026 |

Median best epoch: small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 6.
