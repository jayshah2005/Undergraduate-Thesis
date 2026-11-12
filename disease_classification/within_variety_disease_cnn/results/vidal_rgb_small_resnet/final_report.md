# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| small ResNet, mean pool | 0.376 [0.320, 0.422] | -0.040 [-0.109, +0.031] | 0.450 | 0.420 | 0.500 | 0.440 | 0.562 |
| small ResNet, attention pool | 0.340 [0.291, 0.376] | -0.076 [-0.153, +0.002] | 0.475 | 0.386 | 0.500 | 0.429 | 0.073 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.391, 0.391, 0.367 | 0.383 ± 0.014 |
| small ResNet, attention pool | 0.438, 0.402, 0.378 | 0.406 ± 0.030 |

Median best epoch: small ResNet, mean pool 2, small ResNet, attention pool 3.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| small ResNet, mean pool | 0.484 [0.441, 0.526] | -0.054 [-0.109, +0.000] | 0.558 | 0.558 | 0.475 | 0.508 | 0.500 |
| small ResNet, attention pool | 0.510 [0.459, 0.564] | -0.027 [-0.076, +0.024] | 0.545 | 0.500 | 0.625 | 0.467 | 0.529 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.462, 0.501, 0.451 | 0.472 ± 0.026 |
| small ResNet, attention pool | 0.515, 0.510, 0.462 | 0.496 ± 0.029 |

Median best epoch: small ResNet, mean pool 3, small ResNet, attention pool 10.
