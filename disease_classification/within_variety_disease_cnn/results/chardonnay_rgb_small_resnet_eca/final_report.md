# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| small ResNet + ECA, mean pool | 0.576 [0.527, 0.621] | -0.012 [-0.060, +0.041] | 0.563 | 0.663 | 0.471 | 0.462 | 0.676 |
| small ResNet + ECA, attention pool | 0.581 [0.542, 0.617] | -0.007 [-0.052, +0.038] | 0.542 | 0.500 | 0.500 | 0.487 | 0.624 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.519, 0.594, 0.596 | 0.570 ± 0.044 |
| small ResNet + ECA, attention pool | 0.606, 0.550, 0.581 | 0.579 ± 0.028 |

Median best epoch: small ResNet + ECA, mean pool 4, small ResNet + ECA, attention pool 4.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| small ResNet + ECA, mean pool | 0.587 [0.528, 0.646] | -0.033 [-0.071, +0.005] | 0.561 | 0.533 | 0.727 | 0.583 | 0.463 |
| small ResNet + ECA, attention pool | 0.589 [0.532, 0.647] | -0.030 [-0.072, +0.013] | 0.537 | 0.508 | 0.754 | 0.567 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.578, 0.555, 0.578 | 0.571 ± 0.013 |
| small ResNet + ECA, attention pool | 0.536, 0.584, 0.626 | 0.582 ± 0.045 |

Median best epoch: small ResNet + ECA, mean pool 9, small ResNet + ECA, attention pool 9.
