# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | — | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |
| small ResNet, mean pool | 0.458 [0.440, 0.476] | -0.087 [-0.142, -0.033] | 0.512 | 0.500 | 0.625 | 0.500 | 0.583 |
| small ResNet, attention pool | 0.453 [0.433, 0.473] | -0.091 [-0.144, -0.036] | 0.500 | 0.432 | 0.521 | 0.488 | 0.750 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.431, 0.507, 0.449 | 0.462 ± 0.040 |
| small ResNet, attention pool | 0.473, 0.367, 0.478 | 0.439 ± 0.063 |

Median best epoch: small ResNet, mean pool 13, small ResNet, attention pool 2.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | — | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |
| small ResNet, mean pool | 0.536 [0.484, 0.590] | -0.016 [-0.093, +0.061] | 0.691 | 0.583 | 0.487 | 0.536 | 0.541 |
| small ResNet, attention pool | 0.493 [0.447, 0.538] | -0.060 [-0.133, +0.011] | 0.637 | 0.567 | 0.475 | 0.484 | 0.485 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.439, 0.592, 0.523 | 0.518 ± 0.076 |
| small ResNet, attention pool | 0.483, 0.490, 0.535 | 0.502 ± 0.028 |

Median best epoch: small ResNet, mean pool 17, small ResNet, attention pool 6.
