# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| ResNet-18, mean pool | 0.577 [0.551, 0.600] | +0.033 [-0.003, +0.068] | 0.580 | 0.628 | 0.543 | 0.549 | 0.524 |
| ResNet-18, attention pool | 0.596 [0.568, 0.623] | +0.052 [+0.016, +0.091] | 0.605 | 0.633 | 0.565 | 0.500 | 0.567 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| ResNet-18, mean pool | 0.498 | 0.463 | 0.489 |
| ResNet-18, attention pool | 0.530 | 0.463 | 0.491 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.569, 0.576, 0.579 | 0.574 ± 0.005 |
| ResNet-18, attention pool | 0.578, 0.582, 0.582 | 0.581 ± 0.002 |

Median best epoch: ResNet-18, mean pool 11, ResNet-18, attention pool 10.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | — | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| ResNet-18, mean pool | 0.659 [0.639, 0.678] | +0.018 [-0.006, +0.041] | 0.646 | 0.633 | 0.676 | 0.658 | 0.701 |
| ResNet-18, attention pool | 0.663 [0.644, 0.683] | +0.022 [-0.000, +0.046] | 0.634 | 0.659 | 0.639 | 0.681 | 0.706 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.458 | 0.499 | 0.567 | 0.542 |
| ResNet-18, mean pool | 0.521 | 0.504 | 0.559 | 0.489 |
| ResNet-18, attention pool | 0.491 | 0.551 | 0.535 | 0.529 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.656, 0.634, 0.623 | 0.638 ± 0.017 |
| ResNet-18, attention pool | 0.650, 0.645, 0.668 | 0.654 ± 0.012 |

Median best epoch: ResNet-18, mean pool 19, ResNet-18, attention pool 17.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | — | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| ResNet-18, mean pool | 0.621 [0.597, 0.644] | +0.037 [+0.005, +0.067] | 0.475 | 0.570 | 0.631 | 0.611 | 0.627 |
| ResNet-18, attention pool | 0.604 [0.580, 0.627] | +0.020 [-0.011, +0.049] | 0.539 | 0.530 | 0.632 | 0.600 | 0.581 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.449 | 0.495 | 0.542 | 0.541 |
| ResNet-18, mean pool | 0.518 | 0.555 | 0.579 | 0.547 |
| ResNet-18, attention pool | 0.548 | 0.598 | 0.542 | 0.527 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.621, 0.612, 0.610 | 0.614 ± 0.006 |
| ResNet-18, attention pool | 0.593, 0.592, 0.571 | 0.585 ± 0.012 |

Median best epoch: ResNet-18, mean pool 14, ResNet-18, attention pool 11.
