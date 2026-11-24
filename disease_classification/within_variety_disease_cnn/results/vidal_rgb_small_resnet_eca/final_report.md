# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| small ResNet + ECA, mean pool | 0.464 [0.409, 0.529] | +0.025 [-0.080, +0.133] | 0.500 | 0.458 | 0.500 | 0.351 | 0.458 |
| small ResNet + ECA, attention pool | 0.389 [0.321, 0.460] | -0.050 [-0.139, +0.045] | 0.462 | 0.500 | 0.500 | 0.411 | 0.115 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.443, 0.366, 0.437 | 0.416 ± 0.043 |
| small ResNet + ECA, attention pool | 0.444, 0.426, 0.414 | 0.428 ± 0.015 |

Median best epoch: small ResNet + ECA, mean pool 2, small ResNet + ECA, attention pool 2.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| small ResNet + ECA, mean pool | 0.461 [0.420, 0.497] | -0.091 [-0.146, -0.035] | 0.498 | 0.524 | 0.469 | 0.508 | 0.500 |
| small ResNet + ECA, attention pool | 0.458 [0.414, 0.500] | -0.094 [-0.147, -0.040] | 0.509 | 0.521 | 0.469 | 0.456 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.500, 0.453, 0.479 | 0.477 ± 0.024 |
| small ResNet + ECA, attention pool | 0.469, 0.470, 0.452 | 0.464 ± 0.010 |

Median best epoch: small ResNet + ECA, mean pool 2, small ResNet + ECA, attention pool 2.
