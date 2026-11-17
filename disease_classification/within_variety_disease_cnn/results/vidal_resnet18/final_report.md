# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | — | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |
| ResNet-18, mean pool | 0.396 [0.313, 0.484] | -0.149 [-0.240, -0.051] | 0.294 | 0.561 | 0.583 | 0.494 | 0.375 |
| ResNet-18, attention pool | 0.427 [0.356, 0.504] | -0.118 [-0.196, -0.031] | 0.385 | 0.549 | 0.531 | 0.542 | 0.646 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.424, 0.322, 0.478 | 0.408 ± 0.079 |
| ResNet-18, attention pool | 0.418, 0.389, 0.360 | 0.389 ± 0.029 |

Median best epoch: ResNet-18, mean pool 6, ResNet-18, attention pool 7.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | — | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |
| ResNet-18, mean pool | 0.482 [0.430, 0.534] | -0.070 [-0.149, +0.006] | 0.529 | 0.483 | 0.637 | 0.436 | 0.523 |
| ResNet-18, attention pool | 0.501 [0.449, 0.556] | -0.051 [-0.125, +0.025] | 0.343 | 0.558 | 0.525 | 0.556 | 0.543 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.522, 0.478, 0.447 | 0.482 ± 0.038 |
| ResNet-18, attention pool | 0.516, 0.513, 0.455 | 0.495 ± 0.034 |

Median best epoch: ResNet-18, mean pool 12, ResNet-18, attention pool 8.
