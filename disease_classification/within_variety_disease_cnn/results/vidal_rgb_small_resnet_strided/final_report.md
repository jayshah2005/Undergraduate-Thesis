# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| small ResNet (no max pool), mean pool | 0.431 [0.359, 0.507] | -0.008 [-0.112, +0.098] | 0.487 | 0.508 | 0.500 | 0.458 | 0.510 |
| small ResNet (no max pool), attention pool | 0.411 [0.400, 0.424] | -0.028 [-0.109, +0.060] | 0.500 | 0.500 | 0.500 | 0.500 | 0.125 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.433, 0.392, 0.457 | 0.427 ± 0.033 |
| small ResNet (no max pool), attention pool | 0.409, 0.469, 0.432 | 0.437 ± 0.030 |

Median best epoch: small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 1.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| small ResNet (no max pool), mean pool | 0.479 [0.428, 0.528] | -0.073 [-0.126, -0.023] | 0.535 | 0.517 | 0.462 | 0.491 | 0.514 |
| small ResNet (no max pool), attention pool | 0.498 [0.446, 0.545] | -0.053 [-0.104, -0.001] | 0.518 | 0.540 | 0.535 | 0.508 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.487, 0.511, 0.466 | 0.488 ± 0.022 |
| small ResNet (no max pool), attention pool | 0.477, 0.495, 0.523 | 0.498 ± 0.023 |

Median best epoch: small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 5.
