# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| small ResNet, mean pool | 0.585 [0.556, 0.614] | +0.041 [+0.015, +0.069] | 0.579 | 0.615 | 0.549 | 0.586 | 0.552 |
| small ResNet, attention pool | 0.564 [0.535, 0.590] | +0.020 [-0.006, +0.044] | 0.583 | 0.607 | 0.549 | 0.543 | 0.552 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| small ResNet, mean pool | 0.495 | 0.474 | 0.493 |
| small ResNet, attention pool | 0.483 | 0.412 | 0.496 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.598, 0.544, 0.556 | 0.566 ± 0.029 |
| small ResNet, attention pool | 0.560, 0.554, 0.566 | 0.560 ± 0.006 |

Median best epoch: small ResNet, mean pool 7, small ResNet, attention pool 7.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | — | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| small ResNet, mean pool | 0.608 [0.583, 0.633] | +0.003 [-0.016, +0.023] | 0.622 | 0.568 | 0.648 | 0.582 | 0.657 |
| small ResNet, attention pool | 0.598 [0.572, 0.625] | -0.006 [-0.027, +0.013] | 0.592 | 0.567 | 0.648 | 0.584 | 0.635 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.434 | 0.610 | 0.551 |
| small ResNet, mean pool | 0.473 | 0.583 | 0.552 |
| small ResNet, attention pool | 0.464 | 0.567 | 0.543 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.590, 0.602, 0.607 | 0.600 ± 0.009 |
| small ResNet, attention pool | 0.583, 0.600, 0.601 | 0.595 ± 0.010 |

Median best epoch: small ResNet, mean pool 12, small ResNet, attention pool 6.
