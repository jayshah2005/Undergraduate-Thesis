# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| small ResNet, mean pool | 0.549 [0.533, 0.566] | +0.025 [-0.041, +0.089] | 0.483 | 0.475 | 0.500 | 0.500 | 0.500 |
| small ResNet, attention pool | 0.576 [0.545, 0.610] | +0.052 [-0.017, +0.118] | 0.500 | 0.545 | 0.482 | 0.500 | 0.470 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.565, 0.462, 0.538 | 0.522 ± 0.054 |
| small ResNet, attention pool | 0.585, 0.492, 0.546 | 0.541 ± 0.047 |

Median best epoch: small ResNet, mean pool 3, small ResNet, attention pool 4.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| small ResNet, mean pool | 0.525 [0.480, 0.569] | +0.019 [-0.052, +0.089] | 0.500 | 0.565 | 0.505 | 0.560 | 0.385 |
| small ResNet, attention pool | 0.511 [0.468, 0.553] | +0.005 [-0.069, +0.078] | 0.500 | 0.552 | 0.427 | 0.500 | 0.470 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.485, 0.495, 0.457 | 0.479 ± 0.020 |
| small ResNet, attention pool | 0.488, 0.520, 0.521 | 0.510 ± 0.019 |

Median best epoch: small ResNet, mean pool 6, small ResNet, attention pool 6.
