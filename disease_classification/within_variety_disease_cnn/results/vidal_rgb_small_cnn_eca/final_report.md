# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| CNN + ECA, mean pool | 0.458 [0.402, 0.522] | +0.018 [-0.083, +0.126] | 0.500 | 0.413 | 0.500 | 0.315 | 0.500 |
| CNN + ECA, attention pool | 0.395 [0.319, 0.472] | -0.044 [-0.137, +0.056] | 0.462 | 0.413 | 0.500 | 0.435 | 0.510 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.435, 0.472, 0.475 | 0.461 ± 0.022 |
| CNN + ECA, attention pool | 0.413, 0.385, 0.430 | 0.409 ± 0.022 |

Median best epoch: CNN + ECA, mean pool 1, CNN + ECA, attention pool 1.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| CNN + ECA, mean pool | 0.465 [0.420, 0.508] | -0.087 [-0.142, -0.033] | 0.535 | 0.509 | 0.450 | 0.474 | 0.500 |
| CNN + ECA, attention pool | 0.490 [0.438, 0.540] | -0.062 [-0.115, -0.011] | 0.493 | 0.533 | 0.511 | 0.508 | 0.514 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.490, 0.493, 0.468 | 0.484 ± 0.014 |
| CNN + ECA, attention pool | 0.474, 0.494, 0.526 | 0.498 ± 0.026 |

Median best epoch: CNN + ECA, mean pool 13, CNN + ECA, attention pool 4.
