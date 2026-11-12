# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN, mean pool | 0.370 [0.344, 0.398] | -0.049 [-0.097, +0.002] | 0.500 | 0.519 | 0.304 | 0.400 | 0.517 |
| CNN, attention pool | 0.440 [0.407, 0.472] | +0.022 [-0.028, +0.074] | 0.500 | 0.500 | 0.357 | 0.425 | 0.543 |
| small ResNet, mean pool | 0.535 [0.511, 0.559] | +0.117 [+0.065, +0.170] | 0.466 | 0.500 | 0.454 | 0.525 | 0.500 |
| small ResNet, attention pool | 0.489 [0.453, 0.524] | +0.071 [+0.027, +0.115] | 0.500 | 0.380 | 0.500 | 0.400 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.405, 0.434, 0.365 | 0.401 ± 0.035 |
| CNN, attention pool | 0.509, 0.369, 0.440 | 0.439 ± 0.070 |
| small ResNet, mean pool | 0.557, 0.488, 0.510 | 0.518 ± 0.036 |
| small ResNet, attention pool | 0.531, 0.381, 0.449 | 0.454 ± 0.075 |

Median best epoch: CNN, mean pool 3, CNN, attention pool 2, small ResNet, mean pool 2, small ResNet, attention pool 3.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN, mean pool | 0.525 [0.475, 0.576] | -0.018 [-0.077, +0.037] | 0.649 | 0.524 | 0.502 | 0.500 | 0.397 |
| CNN, attention pool | 0.548 [0.500, 0.595] | +0.005 [-0.060, +0.071] | 0.646 | 0.513 | 0.554 | 0.500 | 0.449 |
| small ResNet, mean pool | 0.542 [0.497, 0.585] | -0.001 [-0.049, +0.047] | 0.528 | 0.552 | 0.554 | 0.600 | 0.383 |
| small ResNet, attention pool | 0.573 [0.533, 0.612] | +0.030 [-0.024, +0.080] | 0.500 | 0.596 | 0.531 | 0.600 | 0.463 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.508, 0.548, 0.524 | 0.527 ± 0.020 |
| CNN, attention pool | 0.543, 0.559, 0.544 | 0.549 ± 0.009 |
| small ResNet, mean pool | 0.535, 0.529, 0.526 | 0.530 ± 0.005 |
| small ResNet, attention pool | 0.547, 0.572, 0.522 | 0.547 ± 0.025 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5, small ResNet, mean pool 7, small ResNet, attention pool 6.
