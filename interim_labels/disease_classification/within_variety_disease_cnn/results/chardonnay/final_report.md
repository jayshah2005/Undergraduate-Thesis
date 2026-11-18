# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN, mean pool | 0.591 [0.555, 0.628] | +0.055 [-0.014, +0.119] | 0.615 | 0.500 | 0.501 | 0.475 | 0.572 |
| CNN, attention pool | 0.559 [0.531, 0.587] | +0.023 [-0.038, +0.079] | 0.500 | 0.500 | 0.471 | 0.475 | 0.522 |
| small ResNet, mean pool | 0.548 [0.517, 0.580] | +0.012 [-0.052, +0.076] | 0.522 | 0.500 | 0.471 | 0.463 | 0.497 |
| small ResNet, attention pool | 0.526 [0.502, 0.550] | -0.011 [-0.075, +0.052] | 0.543 | 0.500 | 0.500 | 0.463 | 0.392 |
| small ResNet (no max pool), mean pool | 0.572 [0.537, 0.608] | +0.036 [-0.032, +0.104] | 0.525 | 0.500 | 0.486 | 0.487 | 0.620 |
| small ResNet (no max pool), attention pool | 0.531 [0.507, 0.555] | -0.006 [-0.069, +0.055] | 0.503 | 0.500 | 0.442 | 0.463 | 0.500 |
| ResNet-18, mean pool | 0.451 [0.406, 0.498] | -0.086 [-0.156, -0.014] | 0.426 | 0.462 | 0.517 | 0.400 | 0.331 |
| ResNet-18, attention pool | 0.518 [0.478, 0.558] | -0.018 [-0.091, +0.051] | 0.448 | 0.462 | 0.531 | 0.512 | 0.498 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.576, 0.546, 0.554 | 0.558 ± 0.016 |
| CNN, attention pool | 0.558, 0.531, 0.572 | 0.554 ± 0.021 |
| small ResNet, mean pool | 0.563, 0.555, 0.531 | 0.550 ± 0.017 |
| small ResNet, attention pool | 0.535, 0.525, 0.539 | 0.533 ± 0.007 |
| small ResNet (no max pool), mean pool | 0.564, 0.547, 0.564 | 0.558 ± 0.010 |
| small ResNet (no max pool), attention pool | 0.531, 0.556, 0.465 | 0.518 ± 0.047 |
| ResNet-18, mean pool | 0.473, 0.455, 0.457 | 0.462 ± 0.010 |
| ResNet-18, attention pool | 0.545, 0.530, 0.470 | 0.515 ± 0.040 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 7, small ResNet, mean pool 9, small ResNet, attention pool 9, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 9, ResNet-18, mean pool 7, ResNet-18, attention pool 10.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN, mean pool | 0.523 [0.475, 0.570] | -0.082 [-0.143, -0.020] | 0.514 | 0.575 | 0.627 | 0.494 | 0.500 |
| CNN, attention pool | 0.519 [0.472, 0.566] | -0.086 [-0.152, -0.023] | 0.537 | 0.600 | 0.668 | 0.500 | 0.500 |
| small ResNet, mean pool | 0.516 [0.465, 0.564] | -0.089 [-0.152, -0.026] | 0.503 | 0.583 | 0.339 | 0.526 | 0.475 |
| small ResNet, attention pool | 0.545 [0.498, 0.591] | -0.060 [-0.121, -0.001] | 0.558 | 0.525 | 0.794 | 0.500 | 0.512 |
| small ResNet (no max pool), mean pool | 0.568 [0.515, 0.622] | -0.037 [-0.102, +0.028] | 0.559 | 0.600 | 0.715 | 0.510 | 0.525 |
| small ResNet (no max pool), attention pool | 0.570 [0.517, 0.623] | -0.035 [-0.101, +0.030] | 0.508 | 0.508 | 0.734 | 0.510 | 0.575 |
| ResNet-18, mean pool | 0.538 [0.485, 0.588] | -0.067 [-0.139, +0.000] | 0.569 | 0.583 | 0.463 | 0.510 | 0.537 |
| ResNet-18, attention pool | 0.497 [0.444, 0.550] | -0.108 [-0.180, -0.037] | 0.490 | 0.575 | 0.404 | 0.494 | 0.575 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.518, 0.563, 0.543 | 0.541 ± 0.022 |
| CNN, attention pool | 0.498, 0.504, 0.541 | 0.514 ± 0.024 |
| small ResNet, mean pool | 0.572, 0.553, 0.533 | 0.553 ± 0.020 |
| small ResNet, attention pool | 0.564, 0.544, 0.556 | 0.555 ± 0.010 |
| small ResNet (no max pool), mean pool | 0.553, 0.530, 0.536 | 0.540 ± 0.012 |
| small ResNet (no max pool), attention pool | 0.539, 0.530, 0.577 | 0.549 ± 0.025 |
| ResNet-18, mean pool | 0.534, 0.505, 0.530 | 0.523 ± 0.016 |
| ResNet-18, attention pool | 0.491, 0.499, 0.495 | 0.495 ± 0.004 |

Median best epoch: CNN, mean pool 5, CNN, attention pool 6, small ResNet, mean pool 7, small ResNet, attention pool 6, small ResNet (no max pool), mean pool 14, small ResNet (no max pool), attention pool 12, ResNet-18, mean pool 8, ResNet-18, attention pool 8.
