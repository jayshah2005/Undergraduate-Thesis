# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| ResNet-18, mean pool | 0.451 [0.406, 0.498] | -0.086 [-0.156, -0.014] | 0.426 | 0.462 | 0.517 | 0.400 | 0.331 |
| ResNet-18, attention pool | 0.518 [0.478, 0.558] | -0.018 [-0.091, +0.051] | 0.448 | 0.462 | 0.531 | 0.512 | 0.498 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.473, 0.455, 0.457 | 0.462 ± 0.010 |
| ResNet-18, attention pool | 0.545, 0.530, 0.470 | 0.515 ± 0.040 |

Median best epoch: ResNet-18, mean pool 7, ResNet-18, attention pool 10.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| ResNet-18, mean pool | 0.538 [0.485, 0.588] | -0.067 [-0.139, +0.000] | 0.569 | 0.583 | 0.463 | 0.510 | 0.537 |
| ResNet-18, attention pool | 0.497 [0.444, 0.550] | -0.108 [-0.180, -0.037] | 0.490 | 0.575 | 0.404 | 0.494 | 0.575 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.534, 0.505, 0.530 | 0.523 ± 0.016 |
| ResNet-18, attention pool | 0.491, 0.499, 0.495 | 0.495 ± 0.004 |

Median best epoch: ResNet-18, mean pool 8, ResNet-18, attention pool 8.
