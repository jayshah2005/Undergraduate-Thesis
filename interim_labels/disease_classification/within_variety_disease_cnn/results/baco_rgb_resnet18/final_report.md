# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| ResNet-18, mean pool | 0.554 [0.505, 0.600] | +0.135 [+0.064, +0.205] | 0.514 | 0.578 | 0.500 | 0.600 | 0.402 |
| ResNet-18, attention pool | 0.582 [0.543, 0.619] | +0.163 [+0.102, +0.224] | 0.535 | 0.565 | 0.500 | 0.475 | 0.450 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.515, 0.516, 0.497 | 0.509 ± 0.011 |
| ResNet-18, attention pool | 0.576, 0.535, 0.539 | 0.550 ± 0.023 |

Median best epoch: ResNet-18, mean pool 7, ResNet-18, attention pool 14.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| ResNet-18, mean pool | 0.515 [0.463, 0.567] | -0.028 [-0.089, +0.032] | 0.559 | 0.494 | 0.502 | 0.500 | 0.486 |
| ResNet-18, attention pool | 0.563 [0.512, 0.615] | +0.020 [-0.037, +0.075] | 0.712 | 0.479 | 0.456 | 0.600 | 0.526 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.476, 0.510, 0.538 | 0.508 ± 0.031 |
| ResNet-18, attention pool | 0.553, 0.549, 0.559 | 0.553 ± 0.005 |

Median best epoch: ResNet-18, mean pool 12, ResNet-18, attention pool 8.
