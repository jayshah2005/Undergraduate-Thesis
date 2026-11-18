# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| small ResNet (no max pool), mean pool | 0.555 [0.525, 0.585] | +0.011 [-0.014, +0.035] | 0.569 | 0.603 | 0.524 | 0.551 | 0.542 |
| small ResNet (no max pool), attention pool | 0.574 [0.546, 0.601] | +0.030 [+0.003, +0.055] | 0.586 | 0.613 | 0.522 | 0.551 | 0.584 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| small ResNet (no max pool), mean pool | 0.468 | 0.427 | 0.487 |
| small ResNet (no max pool), attention pool | 0.499 | 0.423 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.570, 0.508, 0.571 | 0.549 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.569, 0.558, 0.567 | 0.565 ± 0.006 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 7.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | — | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| small ResNet (no max pool), mean pool | 0.665 [0.647, 0.684] | +0.024 [+0.004, +0.045] | 0.671 | 0.653 | 0.659 | 0.688 | 0.668 |
| small ResNet (no max pool), attention pool | 0.656 [0.638, 0.674] | +0.015 [-0.006, +0.035] | 0.641 | 0.669 | 0.654 | 0.671 | 0.653 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.458 | 0.499 | 0.567 | 0.542 |
| small ResNet (no max pool), mean pool | 0.508 | 0.497 | 0.556 | 0.505 |
| small ResNet (no max pool), attention pool | 0.494 | 0.487 | 0.519 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.662, 0.657, 0.660 | 0.660 ± 0.003 |
| small ResNet (no max pool), attention pool | 0.653, 0.649, 0.650 | 0.651 ± 0.003 |

Median best epoch: small ResNet (no max pool), mean pool 18, small ResNet (no max pool), attention pool 19.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | — | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| small ResNet (no max pool), mean pool | 0.577 [0.554, 0.598] | -0.007 [-0.034, +0.018] | 0.450 | 0.537 | 0.593 | 0.612 | 0.604 |
| small ResNet (no max pool), attention pool | 0.599 [0.577, 0.621] | +0.016 [-0.011, +0.040] | 0.564 | 0.553 | 0.603 | 0.638 | 0.612 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.449 | 0.495 | 0.542 | 0.541 |
| small ResNet (no max pool), mean pool | 0.484 | 0.407 | 0.487 | 0.505 |
| small ResNet (no max pool), attention pool | 0.493 | 0.530 | 0.518 | 0.510 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.582, 0.581, 0.588 | 0.584 ± 0.004 |
| small ResNet (no max pool), attention pool | 0.594, 0.584, 0.594 | 0.591 ± 0.006 |

Median best epoch: small ResNet (no max pool), mean pool 16, small ResNet (no max pool), attention pool 11.
