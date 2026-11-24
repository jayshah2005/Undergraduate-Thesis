# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| small ResNet + CBAM, mean pool | 0.556 [0.556, 0.556] | +0.032 [-0.031, +0.092] | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| small ResNet + CBAM, attention pool | 0.559 [0.526, 0.596] | +0.035 [-0.032, +0.097] | 0.578 | 0.406 | 0.500 | 0.513 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.555, 0.472, 0.571 | 0.533 ± 0.053 |
| small ResNet + CBAM, attention pool | 0.543, 0.542, 0.615 | 0.567 ± 0.042 |

Median best epoch: small ResNet + CBAM, mean pool 7, small ResNet + CBAM, attention pool 7.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| small ResNet + CBAM, mean pool | 0.457 [0.422, 0.495] | -0.048 [-0.116, +0.020] | 0.422 | 0.477 | 0.402 | 0.500 | 0.385 |
| small ResNet + CBAM, attention pool | 0.497 [0.447, 0.547] | -0.008 [-0.085, +0.066] | 0.554 | 0.542 | 0.476 | 0.540 | 0.351 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.493, 0.480, 0.531 | 0.502 ± 0.026 |
| small ResNet + CBAM, attention pool | 0.510, 0.468, 0.497 | 0.491 ± 0.021 |

Median best epoch: small ResNet + CBAM, mean pool 6, small ResNet + CBAM, attention pool 5.
