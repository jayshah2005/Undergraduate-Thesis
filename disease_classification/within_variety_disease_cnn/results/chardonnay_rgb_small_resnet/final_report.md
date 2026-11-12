# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| small ResNet, mean pool | 0.580 [0.540, 0.620] | -0.007 [-0.060, +0.049] | 0.575 | 0.500 | 0.516 | 0.513 | 0.551 |
| small ResNet, attention pool | 0.562 [0.520, 0.602] | -0.026 [-0.075, +0.022] | 0.508 | 0.500 | 0.531 | 0.500 | 0.503 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.573, 0.578, 0.579 | 0.577 ± 0.003 |
| small ResNet, attention pool | 0.554, 0.593, 0.548 | 0.565 ± 0.024 |

Median best epoch: small ResNet, mean pool 7, small ResNet, attention pool 10.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| small ResNet, mean pool | 0.578 [0.524, 0.631] | -0.041 [-0.085, -0.003] | 0.558 | 0.525 | 0.740 | 0.510 | 0.463 |
| small ResNet, attention pool | 0.584 [0.529, 0.640] | -0.035 [-0.076, +0.003] | 0.582 | 0.525 | 0.765 | 0.541 | 0.463 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.583, 0.578, 0.555 | 0.572 ± 0.015 |
| small ResNet, attention pool | 0.556, 0.584, 0.565 | 0.568 ± 0.014 |

Median best epoch: small ResNet, mean pool 17, small ResNet, attention pool 7.
