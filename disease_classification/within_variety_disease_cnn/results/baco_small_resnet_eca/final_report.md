# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| small ResNet + ECA, mean pool | 0.491 [0.464, 0.522] | -0.033 [-0.102, +0.034] | 0.524 | 0.489 | 0.500 | 0.500 | 0.500 |
| small ResNet + ECA, attention pool | 0.586 [0.552, 0.617] | +0.061 [-0.008, +0.128] | 0.550 | 0.605 | 0.500 | 0.500 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.456, 0.563, 0.585 | 0.535 ± 0.069 |
| small ResNet + ECA, attention pool | 0.560, 0.500, 0.595 | 0.552 ± 0.048 |

Median best epoch: small ResNet + ECA, mean pool 9, small ResNet + ECA, attention pool 5.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| small ResNet + ECA, mean pool | 0.531 [0.487, 0.579] | +0.026 [-0.050, +0.102] | 0.500 | 0.560 | 0.495 | 0.500 | 0.458 |
| small ResNet + ECA, attention pool | 0.537 [0.494, 0.579] | +0.031 [-0.037, +0.101] | 0.469 | 0.544 | 0.467 | 0.580 | 0.498 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.528, 0.534, 0.496 | 0.519 ± 0.021 |
| small ResNet + ECA, attention pool | 0.478, 0.515, 0.551 | 0.515 ± 0.037 |

Median best epoch: small ResNet + ECA, mean pool 5, small ResNet + ECA, attention pool 4.
