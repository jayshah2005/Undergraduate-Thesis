# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| ResNet-18, mean pool | 0.601 [0.572, 0.627] | +0.024 [-0.020, +0.065] | 0.620 | 0.640 | 0.626 | 0.479 | 0.552 |
| ResNet-18, attention pool | 0.577 [0.549, 0.603] | +0.000 [-0.042, +0.041] | 0.563 | 0.630 | 0.574 | 0.533 | 0.559 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| ResNet-18, mean pool | 0.507 | 0.494 | 0.533 |
| ResNet-18, attention pool | 0.522 | 0.443 | 0.476 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.593, 0.605, 0.554 | 0.584 ± 0.027 |
| ResNet-18, attention pool | 0.541, 0.585, 0.549 | 0.558 ± 0.023 |

Median best epoch: ResNet-18, mean pool 9, ResNet-18, attention pool 12.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | — | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| ResNet-18, mean pool | 0.670 [0.649, 0.692] | +0.033 [+0.004, +0.060] | 0.622 | 0.707 | 0.670 | 0.665 | 0.684 |
| ResNet-18, attention pool | 0.669 [0.649, 0.689] | +0.031 [+0.003, +0.060] | 0.606 | 0.679 | 0.673 | 0.677 | 0.693 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.437 | 0.473 | 0.549 | 0.515 |
| ResNet-18, mean pool | 0.479 | 0.590 | 0.550 | 0.522 |
| ResNet-18, attention pool | 0.471 | 0.642 | 0.540 | 0.516 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.658, 0.655, 0.666 | 0.660 ± 0.006 |
| ResNet-18, attention pool | 0.647, 0.664, 0.641 | 0.651 ± 0.012 |

Median best epoch: ResNet-18, mean pool 22, ResNet-18, attention pool 19.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | — | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |
| ResNet-18, mean pool | 0.577 [0.553, 0.599] | +0.021 [-0.015, +0.057] | 0.582 | 0.619 | 0.600 | 0.592 | 0.512 |
| ResNet-18, attention pool | 0.604 [0.578, 0.630] | +0.049 [+0.011, +0.087] | 0.477 | 0.615 | 0.653 | 0.603 | 0.591 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.431 | 0.435 | 0.491 | 0.609 |
| ResNet-18, mean pool | 0.484 | 0.510 | 0.515 | 0.499 |
| ResNet-18, attention pool | 0.509 | 0.613 | 0.545 | 0.529 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.543, 0.588, 0.580 | 0.570 ± 0.024 |
| ResNet-18, attention pool | 0.613, 0.594, 0.609 | 0.605 ± 0.010 |

Median best epoch: ResNet-18, mean pool 20, ResNet-18, attention pool 16.
