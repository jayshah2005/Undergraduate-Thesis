# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | +0.024 [-0.013, +0.058] | 0.624 | 0.676 | 0.494 | 0.566 | 0.599 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.505 | 0.693 | 0.569 | 0.528 | 0.669 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet (no max pool), mean pool | 0.480 | 0.497 | 0.463 |
| small ResNet (no max pool), attention pool | 0.524 | 0.475 | 0.496 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.561, 0.566, 0.594 | 0.574 ± 0.018 |
| small ResNet (no max pool), attention pool | 0.534, 0.557, 0.595 | 0.562 ± 0.031 |

Median best epoch: small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 11.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| small ResNet (no max pool), mean pool | 0.683 [0.665, 0.700] | +0.036 [+0.012, +0.061] | 0.651 | 0.683 | 0.741 | 0.662 | 0.680 |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | +0.027 [-0.000, +0.053] | 0.677 | 0.657 | 0.689 | 0.675 | 0.686 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet (no max pool), mean pool | 0.494 | 0.531 | 0.554 | 0.501 |
| small ResNet (no max pool), attention pool | 0.466 | 0.563 | 0.527 | 0.530 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.669, 0.654, 0.684 | 0.669 ± 0.015 |
| small ResNet (no max pool), attention pool | 0.651, 0.645, 0.658 | 0.651 ± 0.007 |

Median best epoch: small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 9.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | — | 0.427 | 0.625 | 0.628 | 0.536 | 0.594 |
| small ResNet (no max pool), mean pool | 0.607 [0.580, 0.635] | +0.051 [+0.012, +0.094] | 0.532 | 0.600 | 0.573 | 0.629 | 0.615 |
| small ResNet (no max pool), attention pool | 0.592 [0.567, 0.615] | +0.036 [-0.002, +0.074] | 0.591 | 0.572 | 0.615 | 0.544 | 0.594 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.448 | 0.444 | 0.481 | 0.590 |
| small ResNet (no max pool), mean pool | 0.447 | 0.588 | 0.553 | 0.593 |
| small ResNet (no max pool), attention pool | 0.454 | 0.563 | 0.556 | 0.514 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.580, 0.612, 0.589 | 0.594 ± 0.017 |
| small ResNet (no max pool), attention pool | 0.579, 0.588, 0.564 | 0.577 ± 0.012 |

Median best epoch: small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 8.
