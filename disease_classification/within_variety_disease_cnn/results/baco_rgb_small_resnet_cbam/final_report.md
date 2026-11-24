# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| small ResNet + CBAM, mean pool | 0.580 [0.537, 0.622] | +0.162 [+0.110, +0.212] | 0.641 | 0.506 | 0.500 | 0.513 | 0.567 |
| small ResNet + CBAM, attention pool | 0.484 [0.426, 0.540] | +0.066 [+0.008, +0.121] | 0.443 | 0.505 | 0.568 | 0.388 | 0.504 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.552, 0.513, 0.459 | 0.508 ± 0.047 |
| small ResNet + CBAM, attention pool | 0.411, 0.514, 0.514 | 0.480 ± 0.059 |

Median best epoch: small ResNet + CBAM, mean pool 3, small ResNet + CBAM, attention pool 6.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| small ResNet + CBAM, mean pool | 0.530 [0.482, 0.573] | -0.014 [-0.069, +0.040] | 0.545 | 0.623 | 0.509 | 0.520 | 0.359 |
| small ResNet + CBAM, attention pool | 0.554 [0.506, 0.601] | +0.011 [-0.040, +0.059] | 0.552 | 0.614 | 0.509 | 0.580 | 0.425 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.507, 0.522, 0.575 | 0.535 ± 0.035 |
| small ResNet + CBAM, attention pool | 0.531, 0.540, 0.580 | 0.550 ± 0.026 |

Median best epoch: small ResNet + CBAM, mean pool 6, small ResNet + CBAM, attention pool 7.
