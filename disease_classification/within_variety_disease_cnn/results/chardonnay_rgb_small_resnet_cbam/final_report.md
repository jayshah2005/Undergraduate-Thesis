# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| small ResNet + CBAM, mean pool | 0.567 [0.520, 0.615] | -0.020 [-0.073, +0.033] | 0.601 | 0.484 | 0.518 | 0.487 | 0.510 |
| small ResNet + CBAM, attention pool | 0.540 [0.492, 0.587] | -0.048 [-0.101, +0.005] | 0.591 | 0.651 | 0.456 | 0.487 | 0.510 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.546, 0.563, 0.566 | 0.558 ± 0.011 |
| small ResNet + CBAM, attention pool | 0.499, 0.530, 0.560 | 0.530 ± 0.031 |

Median best epoch: small ResNet + CBAM, mean pool 7, small ResNet + CBAM, attention pool 5.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| small ResNet + CBAM, mean pool | 0.569 [0.513, 0.628] | -0.050 [-0.091, -0.010] | 0.540 | 0.508 | 0.700 | 0.573 | 0.463 |
| small ResNet + CBAM, attention pool | 0.586 [0.527, 0.645] | -0.034 [-0.074, +0.008] | 0.540 | 0.517 | 0.779 | 0.583 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.537, 0.550, 0.594 | 0.560 ± 0.030 |
| small ResNet + CBAM, attention pool | 0.574, 0.573, 0.589 | 0.579 ± 0.009 |

Median best epoch: small ResNet + CBAM, mean pool 8, small ResNet + CBAM, attention pool 12.
