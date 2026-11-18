# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

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

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | — | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| ResNet-18, mean pool | 0.590 [0.567, 0.614] | -0.007 [-0.041, +0.026] | 0.564 | 0.580 | 0.652 | 0.571 | 0.624 |
| ResNet-18, attention pool | 0.577 [0.552, 0.601] | -0.020 [-0.054, +0.014] | 0.512 | 0.593 | 0.664 | 0.567 | 0.580 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.460 | 0.548 | 0.546 |
| ResNet-18, mean pool | 0.493 | 0.533 | 0.539 |
| ResNet-18, attention pool | 0.469 | 0.526 | 0.540 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.585, 0.590, 0.556 | 0.577 ± 0.018 |
| ResNet-18, attention pool | 0.573, 0.557, 0.553 | 0.561 ± 0.010 |

Median best epoch: ResNet-18, mean pool 12, ResNet-18, attention pool 14.
