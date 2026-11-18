# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| small ResNet, mean pool | 0.513 [0.441, 0.589] | -0.043 [-0.121, +0.031] | 0.568 | 0.375 | 0.628 | 0.491 | 0.490 |
| small ResNet, attention pool | 0.432 [0.377, 0.488] | -0.125 [-0.213, -0.038] | 0.472 | 0.386 | 0.521 | 0.488 | 0.740 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.507, 0.580, 0.526 | 0.538 ± 0.038 |
| small ResNet, attention pool | 0.458, 0.443, 0.431 | 0.444 ± 0.013 |

Median best epoch: small ResNet, mean pool 14, small ResNet, attention pool 2.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| small ResNet, mean pool | 0.503 [0.450, 0.556] | -0.044 [-0.119, +0.029] | 0.612 | 0.564 | 0.505 | 0.450 | 0.512 |
| small ResNet, attention pool | 0.509 [0.456, 0.560] | -0.038 [-0.115, +0.037] | 0.695 | 0.540 | 0.505 | 0.484 | 0.481 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.498, 0.565, 0.495 | 0.519 ± 0.040 |
| small ResNet, attention pool | 0.505, 0.513, 0.526 | 0.514 ± 0.010 |

Median best epoch: small ResNet, mean pool 8, small ResNet, attention pool 9.
