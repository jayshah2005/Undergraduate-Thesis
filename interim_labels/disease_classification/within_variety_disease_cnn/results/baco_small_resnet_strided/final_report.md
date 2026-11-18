# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| small ResNet (no max pool), mean pool | 0.506 [0.467, 0.545] | -0.018 [-0.089, +0.050] | 0.548 | 0.453 | 0.500 | 0.463 | 0.383 |
| small ResNet (no max pool), attention pool | 0.605 [0.571, 0.638] | +0.081 [+0.009, +0.150] | 0.500 | 0.644 | 0.525 | 0.513 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.547, 0.507, 0.429 | 0.494 ± 0.060 |
| small ResNet (no max pool), attention pool | 0.526, 0.505, 0.567 | 0.533 ± 0.032 |

Median best epoch: small ResNet (no max pool), mean pool 6, small ResNet (no max pool), attention pool 10.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| small ResNet (no max pool), mean pool | 0.533 [0.487, 0.577] | +0.027 [-0.044, +0.100] | 0.500 | 0.505 | 0.554 | 0.540 | 0.415 |
| small ResNet (no max pool), attention pool | 0.518 [0.485, 0.549] | +0.012 [-0.054, +0.079] | 0.500 | 0.500 | 0.490 | 0.500 | 0.455 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.557, 0.512, 0.531 | 0.533 ± 0.022 |
| small ResNet (no max pool), attention pool | 0.517, 0.561, 0.526 | 0.535 ± 0.023 |

Median best epoch: small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 3.
