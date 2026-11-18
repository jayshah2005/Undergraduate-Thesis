# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | +0.044 [+0.007, +0.080] | 0.557 | 0.628 | 0.540 | 0.509 | 0.571 |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | +0.068 [+0.033, +0.105] | 0.595 | 0.633 | 0.561 | 0.493 | 0.637 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| ResNet-18, mean pool | 0.523 | 0.509 | 0.459 |
| ResNet-18, attention pool | 0.549 | 0.471 | 0.520 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.591, 0.599, 0.570 | 0.587 ± 0.015 |
| ResNet-18, attention pool | 0.597, 0.592, 0.583 | 0.591 ± 0.007 |

Median best epoch: ResNet-18, mean pool 15, ResNet-18, attention pool 10.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| ResNet-18, mean pool | 0.665 [0.643, 0.685] | +0.020 [-0.004, +0.044] | 0.627 | 0.641 | 0.711 | 0.663 | 0.679 |
| ResNet-18, attention pool | 0.675 [0.655, 0.694] | +0.031 [+0.008, +0.054] | 0.650 | 0.677 | 0.671 | 0.679 | 0.698 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| ResNet-18, mean pool | 0.480 | 0.516 | 0.537 | 0.525 |
| ResNet-18, attention pool | 0.520 | 0.527 | 0.564 | 0.487 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.659, 0.659, 0.654 | 0.657 ± 0.003 |
| ResNet-18, attention pool | 0.649, 0.660, 0.654 | 0.654 ± 0.006 |

Median best epoch: ResNet-18, mean pool 15, ResNet-18, attention pool 20.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | — | 0.576 | 0.542 | 0.655 | 0.591 | 0.577 |
| ResNet-18, mean pool | 0.640 [0.617, 0.662] | +0.058 [+0.029, +0.087] | 0.652 | 0.566 | 0.614 | 0.615 | 0.630 |
| ResNet-18, attention pool | 0.620 [0.595, 0.646] | +0.037 [+0.008, +0.066] | 0.664 | 0.558 | 0.611 | 0.555 | 0.639 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.440 | 0.497 | 0.542 | 0.532 |
| ResNet-18, mean pool | 0.600 | 0.677 | 0.556 | 0.491 |
| ResNet-18, attention pool | 0.564 | 0.592 | 0.565 | 0.508 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.599, 0.629, 0.632 | 0.620 ± 0.018 |
| ResNet-18, attention pool | 0.636, 0.608, 0.570 | 0.605 ± 0.033 |

Median best epoch: ResNet-18, mean pool 22, ResNet-18, attention pool 11.
