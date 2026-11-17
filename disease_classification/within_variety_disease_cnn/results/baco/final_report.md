# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| CNN, mean pool | 0.546 [0.522, 0.571] | +0.021 [-0.043, +0.084] | 0.524 | 0.500 | 0.525 | 0.500 | 0.500 |
| CNN, attention pool | 0.529 [0.502, 0.556] | +0.005 [-0.060, +0.068] | 0.500 | 0.494 | 0.500 | 0.463 | 0.500 |
| small ResNet, mean pool | 0.549 [0.533, 0.566] | +0.025 [-0.041, +0.089] | 0.483 | 0.475 | 0.500 | 0.500 | 0.500 |
| small ResNet, attention pool | 0.576 [0.545, 0.610] | +0.052 [-0.017, +0.118] | 0.500 | 0.545 | 0.482 | 0.500 | 0.470 |
| small ResNet (no max pool), mean pool | 0.506 [0.467, 0.545] | -0.018 [-0.089, +0.050] | 0.548 | 0.453 | 0.500 | 0.463 | 0.383 |
| small ResNet (no max pool), attention pool | 0.605 [0.571, 0.638] | +0.081 [+0.009, +0.150] | 0.500 | 0.644 | 0.525 | 0.513 | 0.500 |
| ResNet-18, mean pool | 0.566 [0.526, 0.608] | +0.042 [-0.034, +0.120] | 0.526 | 0.507 | 0.500 | 0.513 | 0.446 |
| ResNet-18, attention pool | 0.533 [0.487, 0.578] | +0.009 [-0.068, +0.086] | 0.583 | 0.457 | 0.500 | 0.488 | 0.398 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.555, 0.576, 0.518 | 0.550 ± 0.029 |
| CNN, attention pool | 0.505, 0.577, 0.554 | 0.546 ± 0.037 |
| small ResNet, mean pool | 0.565, 0.462, 0.538 | 0.522 ± 0.054 |
| small ResNet, attention pool | 0.585, 0.492, 0.546 | 0.541 ± 0.047 |
| small ResNet (no max pool), mean pool | 0.547, 0.507, 0.429 | 0.494 ± 0.060 |
| small ResNet (no max pool), attention pool | 0.526, 0.505, 0.567 | 0.533 ± 0.032 |
| ResNet-18, mean pool | 0.523, 0.520, 0.523 | 0.522 ± 0.002 |
| ResNet-18, attention pool | 0.487, 0.541, 0.510 | 0.513 ± 0.027 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 10, small ResNet, mean pool 3, small ResNet, attention pool 4, small ResNet (no max pool), mean pool 6, small ResNet (no max pool), attention pool 10, ResNet-18, mean pool 9, ResNet-18, attention pool 9.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| CNN, mean pool | 0.504 [0.456, 0.550] | -0.001 [-0.074, +0.074] | 0.519 | 0.500 | 0.451 | 0.600 | 0.471 |
| CNN, attention pool | 0.505 [0.465, 0.547] | -0.001 [-0.068, +0.069] | 0.552 | 0.409 | 0.500 | 0.500 | 0.441 |
| small ResNet, mean pool | 0.525 [0.480, 0.569] | +0.019 [-0.052, +0.089] | 0.500 | 0.565 | 0.505 | 0.560 | 0.385 |
| small ResNet, attention pool | 0.511 [0.468, 0.553] | +0.005 [-0.069, +0.078] | 0.500 | 0.552 | 0.427 | 0.500 | 0.470 |
| small ResNet (no max pool), mean pool | 0.533 [0.487, 0.577] | +0.027 [-0.044, +0.100] | 0.500 | 0.505 | 0.554 | 0.540 | 0.415 |
| small ResNet (no max pool), attention pool | 0.518 [0.485, 0.549] | +0.012 [-0.054, +0.079] | 0.500 | 0.500 | 0.490 | 0.500 | 0.455 |
| ResNet-18, mean pool | 0.505 [0.448, 0.559] | -0.001 [-0.075, +0.080] | 0.451 | 0.463 | 0.566 | 0.540 | 0.441 |
| ResNet-18, attention pool | 0.471 [0.425, 0.522] | -0.034 [-0.101, +0.040] | 0.464 | 0.518 | 0.444 | 0.500 | 0.389 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.471, 0.496, 0.491 | 0.486 ± 0.013 |
| CNN, attention pool | 0.511, 0.514, 0.523 | 0.516 ± 0.006 |
| small ResNet, mean pool | 0.485, 0.495, 0.457 | 0.479 ± 0.020 |
| small ResNet, attention pool | 0.488, 0.520, 0.521 | 0.510 ± 0.019 |
| small ResNet (no max pool), mean pool | 0.557, 0.512, 0.531 | 0.533 ± 0.022 |
| small ResNet (no max pool), attention pool | 0.517, 0.561, 0.526 | 0.535 ± 0.023 |
| ResNet-18, mean pool | 0.522, 0.493, 0.506 | 0.507 ± 0.015 |
| ResNet-18, attention pool | 0.502, 0.500, 0.489 | 0.497 ± 0.007 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5, small ResNet, mean pool 6, small ResNet, attention pool 6, small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 3, ResNet-18, mean pool 7, ResNet-18, attention pool 14.
