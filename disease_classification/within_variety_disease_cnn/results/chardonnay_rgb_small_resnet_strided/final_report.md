# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| small ResNet (no max pool), mean pool | 0.575 [0.535, 0.615] | -0.013 [-0.065, +0.039] | 0.678 | 0.500 | 0.486 | 0.525 | 0.482 |
| small ResNet (no max pool), attention pool | 0.572 [0.536, 0.608] | -0.015 [-0.060, +0.029] | 0.486 | 0.500 | 0.471 | 0.500 | 0.628 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.563, 0.575, 0.602 | 0.580 ± 0.020 |
| small ResNet (no max pool), attention pool | 0.574, 0.590, 0.557 | 0.574 ± 0.016 |

Median best epoch: small ResNet (no max pool), mean pool 7, small ResNet (no max pool), attention pool 6.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| small ResNet (no max pool), mean pool | 0.584 [0.526, 0.641] | -0.036 [-0.079, +0.006] | 0.540 | 0.500 | 0.740 | 0.573 | 0.512 |
| small ResNet (no max pool), attention pool | 0.572 [0.515, 0.627] | -0.048 [-0.088, -0.009] | 0.540 | 0.508 | 0.738 | 0.541 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.597, 0.556, 0.568 | 0.574 ± 0.021 |
| small ResNet (no max pool), attention pool | 0.595, 0.566, 0.599 | 0.587 ± 0.018 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 9.
