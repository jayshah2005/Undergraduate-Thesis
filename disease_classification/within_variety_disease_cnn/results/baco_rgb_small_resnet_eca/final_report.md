# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| small ResNet + ECA, mean pool | 0.490 [0.444, 0.535] | +0.072 [+0.009, +0.136] | 0.479 | 0.512 | 0.489 | 0.575 | 0.500 |
| small ResNet + ECA, attention pool | 0.454 [0.409, 0.500] | +0.036 [-0.014, +0.088] | 0.415 | 0.546 | 0.500 | 0.375 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.538, 0.424, 0.449 | 0.470 ± 0.060 |
| small ResNet + ECA, attention pool | 0.487, 0.426, 0.526 | 0.480 ± 0.050 |

Median best epoch: small ResNet + ECA, mean pool 3, small ResNet + ECA, attention pool 3.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| small ResNet + ECA, mean pool | 0.570 [0.527, 0.611] | +0.027 [-0.031, +0.084] | 0.641 | 0.591 | 0.554 | 0.500 | 0.444 |
| small ResNet + ECA, attention pool | 0.554 [0.508, 0.600] | +0.011 [-0.033, +0.055] | 0.500 | 0.614 | 0.486 | 0.600 | 0.441 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.520, 0.561, 0.525 | 0.535 ± 0.023 |
| small ResNet + ECA, attention pool | 0.534, 0.527, 0.546 | 0.536 ± 0.009 |

Median best epoch: small ResNet + ECA, mean pool 5, small ResNet + ECA, attention pool 6.
