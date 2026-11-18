# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| CNN, mean pool | 0.474 [0.419, 0.525] | -0.082 [-0.173, +0.002] | 0.600 | 0.466 | 0.511 | 0.488 | 0.760 |
| CNN, attention pool | 0.447 [0.387, 0.510] | -0.110 [-0.182, -0.035] | 0.478 | 0.447 | 0.543 | 0.488 | 0.750 |
| small ResNet, mean pool | 0.513 [0.441, 0.589] | -0.043 [-0.121, +0.031] | 0.568 | 0.375 | 0.628 | 0.491 | 0.490 |
| small ResNet, attention pool | 0.432 [0.377, 0.488] | -0.125 [-0.213, -0.038] | 0.472 | 0.386 | 0.521 | 0.488 | 0.740 |
| small ResNet (no max pool), mean pool | 0.448 [0.390, 0.501] | -0.109 [-0.203, -0.025] | 0.611 | 0.409 | 0.670 | 0.500 | 0.510 |
| small ResNet (no max pool), attention pool | 0.488 [0.437, 0.536] | -0.068 [-0.135, -0.001] | 0.558 | 0.458 | 0.681 | 0.500 | 0.562 |
| ResNet-18, mean pool | 0.440 [0.365, 0.518] | -0.117 [-0.219, -0.014] | 0.575 | 0.538 | 0.585 | 0.491 | 0.135 |
| ResNet-18, attention pool | 0.440 [0.368, 0.517] | -0.117 [-0.216, -0.019] | 0.470 | 0.504 | 0.532 | 0.539 | 0.594 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.479, 0.424, 0.397 | 0.434 ± 0.042 |
| CNN, attention pool | 0.451, 0.382, 0.486 | 0.439 ± 0.053 |
| small ResNet, mean pool | 0.507, 0.580, 0.526 | 0.538 ± 0.038 |
| small ResNet, attention pool | 0.458, 0.443, 0.431 | 0.444 ± 0.013 |
| small ResNet (no max pool), mean pool | 0.463, 0.445, 0.389 | 0.432 ± 0.038 |
| small ResNet (no max pool), attention pool | 0.458, 0.474, 0.472 | 0.468 ± 0.009 |
| ResNet-18, mean pool | 0.404, 0.414, 0.478 | 0.432 ± 0.040 |
| ResNet-18, attention pool | 0.430, 0.409, 0.435 | 0.425 ± 0.014 |

Median best epoch: CNN, mean pool 7, CNN, attention pool 10, small ResNet, mean pool 14, small ResNet, attention pool 2, small ResNet (no max pool), mean pool 13, small ResNet (no max pool), attention pool 4, ResNet-18, mean pool 6, ResNet-18, attention pool 7.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| CNN, mean pool | 0.557 [0.498, 0.612] | +0.009 [-0.071, +0.090] | 0.640 | 0.589 | 0.542 | 0.525 | 0.582 |
| CNN, attention pool | 0.492 [0.442, 0.541] | -0.055 [-0.127, +0.016] | 0.553 | 0.521 | 0.481 | 0.525 | 0.555 |
| small ResNet, mean pool | 0.503 [0.450, 0.556] | -0.044 [-0.119, +0.029] | 0.612 | 0.564 | 0.505 | 0.450 | 0.512 |
| small ResNet, attention pool | 0.509 [0.456, 0.560] | -0.038 [-0.115, +0.037] | 0.695 | 0.540 | 0.505 | 0.484 | 0.481 |
| small ResNet (no max pool), mean pool | 0.509 [0.453, 0.562] | -0.038 [-0.109, +0.035] | 0.586 | 0.620 | 0.505 | 0.467 | 0.498 |
| small ResNet (no max pool), attention pool | 0.519 [0.467, 0.570] | -0.028 [-0.098, +0.044] | 0.671 | 0.604 | 0.469 | 0.467 | 0.541 |
| ResNet-18, mean pool | 0.533 [0.479, 0.586] | -0.015 [-0.095, +0.065] | 0.487 | 0.592 | 0.518 | 0.580 | 0.539 |
| ResNet-18, attention pool | 0.510 [0.462, 0.559] | -0.038 [-0.117, +0.043] | 0.590 | 0.500 | 0.518 | 0.575 | 0.586 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.524, 0.533, 0.598 | 0.552 ± 0.040 |
| CNN, attention pool | 0.525, 0.454, 0.496 | 0.492 ± 0.036 |
| small ResNet, mean pool | 0.498, 0.565, 0.495 | 0.519 ± 0.040 |
| small ResNet, attention pool | 0.505, 0.513, 0.526 | 0.514 ± 0.010 |
| small ResNet (no max pool), mean pool | 0.514, 0.528, 0.522 | 0.521 ± 0.007 |
| small ResNet (no max pool), attention pool | 0.539, 0.504, 0.465 | 0.503 ± 0.037 |
| ResNet-18, mean pool | 0.547, 0.471, 0.495 | 0.504 ± 0.038 |
| ResNet-18, attention pool | 0.539, 0.488, 0.496 | 0.508 ± 0.027 |

Median best epoch: CNN, mean pool 10, CNN, attention pool 5, small ResNet, mean pool 8, small ResNet, attention pool 9, small ResNet (no max pool), mean pool 4, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 9, ResNet-18, attention pool 10.
