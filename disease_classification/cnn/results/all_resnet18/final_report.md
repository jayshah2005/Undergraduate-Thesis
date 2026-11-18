# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | +0.035 [-0.008, +0.075] | 0.541 | 0.652 | 0.623 | 0.561 | 0.598 |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | +0.002 [-0.042, +0.043] | 0.567 | 0.629 | 0.571 | 0.515 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| ResNet-18, mean pool | 0.548 | 0.502 | 0.493 |
| ResNet-18, attention pool | 0.507 | 0.419 | 0.514 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.594, 0.609, 0.539 | 0.581 ± 0.037 |
| ResNet-18, attention pool | 0.540, 0.589, 0.570 | 0.566 ± 0.025 |

Median best epoch: ResNet-18, mean pool 7, ResNet-18, attention pool 11.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| ResNet-18, mean pool | 0.672 [0.647, 0.693] | +0.025 [-0.004, +0.053] | 0.640 | 0.684 | 0.644 | 0.681 | 0.705 |
| ResNet-18, attention pool | 0.667 [0.643, 0.689] | +0.020 [-0.010, +0.049] | 0.627 | 0.671 | 0.658 | 0.646 | 0.726 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| ResNet-18, mean pool | 0.460 | 0.701 | 0.545 | 0.529 |
| ResNet-18, attention pool | 0.472 | 0.537 | 0.541 | 0.518 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.654, 0.635, 0.672 | 0.654 ± 0.018 |
| ResNet-18, attention pool | 0.667, 0.679, 0.643 | 0.663 ± 0.018 |

Median best epoch: ResNet-18, mean pool 20, ResNet-18, attention pool 18.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | — | 0.427 | 0.625 | 0.628 | 0.536 | 0.594 |
| ResNet-18, mean pool | 0.602 [0.575, 0.630] | +0.046 [+0.006, +0.087] | 0.500 | 0.660 | 0.595 | 0.572 | 0.638 |
| ResNet-18, attention pool | 0.621 [0.594, 0.646] | +0.065 [+0.026, +0.103] | 0.594 | 0.624 | 0.592 | 0.604 | 0.583 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.448 | 0.444 | 0.481 | 0.590 |
| ResNet-18, mean pool | 0.467 | 0.555 | 0.575 | 0.549 |
| ResNet-18, attention pool | 0.554 | 0.543 | 0.561 | 0.549 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.598, 0.619, 0.555 | 0.590 ± 0.033 |
| ResNet-18, attention pool | 0.628, 0.607, 0.596 | 0.610 ± 0.016 |

Median best epoch: ResNet-18, mean pool 15, ResNet-18, attention pool 16.
