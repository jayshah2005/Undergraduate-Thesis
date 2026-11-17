# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| ResNet-18, mean pool | 0.538 [0.486, 0.586] | -0.050 [-0.122, +0.018] | 0.625 | 0.510 | 0.503 | 0.412 | 0.557 |
| ResNet-18, attention pool | 0.485 [0.433, 0.538] | -0.102 [-0.173, -0.034] | 0.482 | 0.571 | 0.546 | 0.425 | 0.471 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.507, 0.522, 0.545 | 0.525 ± 0.019 |
| ResNet-18, attention pool | 0.441, 0.483, 0.473 | 0.466 ± 0.022 |

Median best epoch: ResNet-18, mean pool 14, ResNet-18, attention pool 15.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| ResNet-18, mean pool | 0.503 [0.443, 0.563] | -0.116 [-0.171, -0.062] | 0.417 | 0.542 | 0.526 | 0.498 | 0.537 |
| ResNet-18, attention pool | 0.545 [0.487, 0.597] | -0.075 [-0.129, -0.022] | 0.454 | 0.483 | 0.630 | 0.504 | 0.650 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.532, 0.504, 0.541 | 0.526 ± 0.019 |
| ResNet-18, attention pool | 0.553, 0.566, 0.510 | 0.543 ± 0.030 |

Median best epoch: ResNet-18, mean pool 13, ResNet-18, attention pool 11.
