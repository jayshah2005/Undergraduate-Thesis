# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

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

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | — | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| ResNet-18, mean pool | 0.562 [0.536, 0.589] | -0.042 [-0.074, -0.010] | 0.489 | 0.595 | 0.589 | 0.544 | 0.603 |
| ResNet-18, attention pool | 0.587 [0.562, 0.613] | -0.017 [-0.048, +0.011] | 0.541 | 0.621 | 0.679 | 0.539 | 0.562 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.434 | 0.610 | 0.551 |
| ResNet-18, mean pool | 0.493 | 0.494 | 0.522 |
| ResNet-18, attention pool | 0.527 | 0.514 | 0.542 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.567, 0.531, 0.510 | 0.536 ± 0.029 |
| ResNet-18, attention pool | 0.559, 0.593, 0.582 | 0.578 ± 0.018 |

Median best epoch: ResNet-18, mean pool 13, ResNet-18, attention pool 13.
