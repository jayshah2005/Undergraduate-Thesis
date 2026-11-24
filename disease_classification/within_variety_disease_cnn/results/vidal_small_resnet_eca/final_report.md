# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| small ResNet + ECA, mean pool | 0.419 [0.348, 0.488] | -0.138 [-0.237, -0.042] | 0.390 | 0.485 | 0.660 | 0.488 | 0.500 |
| small ResNet + ECA, attention pool | 0.499 [0.453, 0.552] | -0.058 [-0.142, +0.028] | 0.500 | 0.371 | 0.649 | 0.500 | 0.479 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.411, 0.449, 0.496 | 0.452 ± 0.043 |
| small ResNet + ECA, attention pool | 0.502, 0.456, 0.470 | 0.476 ± 0.023 |

Median best epoch: small ResNet + ECA, mean pool 11, small ResNet + ECA, attention pool 5.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| small ResNet + ECA, mean pool | 0.547 [0.496, 0.596] | -0.001 [-0.073, +0.071] | 0.697 | 0.568 | 0.547 | 0.498 | 0.526 |
| small ResNet + ECA, attention pool | 0.485 [0.438, 0.530] | -0.062 [-0.129, +0.006] | 0.645 | 0.500 | 0.481 | 0.484 | 0.493 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.495, 0.578, 0.548 | 0.540 ± 0.042 |
| small ResNet + ECA, attention pool | 0.497, 0.520, 0.474 | 0.497 ± 0.023 |

Median best epoch: small ResNet + ECA, mean pool 5, small ResNet + ECA, attention pool 8.
