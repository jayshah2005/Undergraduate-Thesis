# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | +0.052 [+0.022, +0.083] | 0.577 | 0.615 | 0.545 | 0.581 | 0.527 |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | +0.022 [-0.002, +0.046] | 0.568 | 0.607 | 0.544 | 0.584 | 0.533 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| small ResNet, mean pool | 0.524 | 0.525 | 0.446 |
| small ResNet, attention pool | 0.477 | 0.440 | 0.488 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.594, 0.547, 0.567 | 0.569 ± 0.023 |
| small ResNet, attention pool | 0.572, 0.546, 0.567 | 0.561 ± 0.014 |

Median best epoch: small ResNet, mean pool 7, small ResNet, attention pool 4.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| small ResNet, mean pool | 0.604 [0.579, 0.630] | -0.005 [-0.026, +0.016] | 0.595 | 0.568 | 0.662 | 0.569 | 0.655 |
| small ResNet, attention pool | 0.596 [0.568, 0.622] | -0.014 [-0.034, +0.008] | 0.594 | 0.573 | 0.668 | 0.549 | 0.633 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| small ResNet, mean pool | 0.477 | 0.568 | 0.539 |
| small ResNet, attention pool | 0.480 | 0.559 | 0.530 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.590, 0.599, 0.600 | 0.596 ± 0.005 |
| small ResNet, attention pool | 0.597, 0.589, 0.605 | 0.597 ± 0.008 |

Median best epoch: small ResNet, mean pool 6, small ResNet, attention pool 6.
