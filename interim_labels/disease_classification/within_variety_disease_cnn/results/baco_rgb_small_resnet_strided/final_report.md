# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| small ResNet (no max pool), mean pool | 0.497 [0.459, 0.532] | +0.079 [+0.025, +0.131] | 0.417 | 0.419 | 0.500 | 0.500 | 0.500 |
| small ResNet (no max pool), attention pool | 0.534 [0.505, 0.561] | +0.115 [+0.065, +0.166] | 0.500 | 0.515 | 0.500 | 0.500 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.523, 0.470, 0.432 | 0.475 ± 0.046 |
| small ResNet (no max pool), attention pool | 0.585, 0.481, 0.412 | 0.492 ± 0.087 |

Median best epoch: small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 2.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| small ResNet (no max pool), mean pool | 0.563 [0.521, 0.607] | +0.020 [-0.038, +0.081] | 0.500 | 0.515 | 0.509 | 0.600 | 0.566 |
| small ResNet (no max pool), attention pool | 0.534 [0.492, 0.576] | -0.009 [-0.054, +0.034] | 0.500 | 0.570 | 0.509 | 0.580 | 0.378 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.577, 0.500, 0.552 | 0.543 ± 0.039 |
| small ResNet (no max pool), attention pool | 0.520, 0.549, 0.514 | 0.528 ± 0.019 |

Median best epoch: small ResNet (no max pool), mean pool 7, small ResNet (no max pool), attention pool 7.
