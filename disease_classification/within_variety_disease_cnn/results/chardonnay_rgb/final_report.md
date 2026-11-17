# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN, mean pool | 0.531 [0.487, 0.575] | -0.057 [-0.105, -0.009] | 0.567 | 0.500 | 0.486 | 0.475 | 0.465 |
| CNN, attention pool | 0.569 [0.525, 0.613] | -0.019 [-0.068, +0.034] | 0.470 | 0.598 | 0.500 | 0.525 | 0.607 |
| small ResNet, mean pool | 0.580 [0.540, 0.620] | -0.007 [-0.060, +0.049] | 0.575 | 0.500 | 0.516 | 0.513 | 0.551 |
| small ResNet, attention pool | 0.562 [0.520, 0.602] | -0.026 [-0.075, +0.022] | 0.508 | 0.500 | 0.531 | 0.500 | 0.503 |
| small ResNet (no max pool), mean pool | 0.575 [0.535, 0.615] | -0.013 [-0.065, +0.039] | 0.678 | 0.500 | 0.486 | 0.525 | 0.482 |
| small ResNet (no max pool), attention pool | 0.572 [0.536, 0.608] | -0.015 [-0.060, +0.029] | 0.486 | 0.500 | 0.471 | 0.500 | 0.628 |
| ResNet-18, mean pool | 0.538 [0.486, 0.586] | -0.050 [-0.122, +0.018] | 0.625 | 0.510 | 0.503 | 0.412 | 0.557 |
| ResNet-18, attention pool | 0.485 [0.433, 0.538] | -0.102 [-0.173, -0.034] | 0.482 | 0.571 | 0.546 | 0.425 | 0.471 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.529, 0.521, 0.522 | 0.524 ± 0.004 |
| CNN, attention pool | 0.562, 0.572, 0.498 | 0.544 ± 0.040 |
| small ResNet, mean pool | 0.573, 0.578, 0.579 | 0.577 ± 0.003 |
| small ResNet, attention pool | 0.554, 0.593, 0.548 | 0.565 ± 0.024 |
| small ResNet (no max pool), mean pool | 0.563, 0.575, 0.602 | 0.580 ± 0.020 |
| small ResNet (no max pool), attention pool | 0.574, 0.590, 0.557 | 0.574 ± 0.016 |
| ResNet-18, mean pool | 0.507, 0.522, 0.545 | 0.525 ± 0.019 |
| ResNet-18, attention pool | 0.441, 0.483, 0.473 | 0.466 ± 0.022 |

Median best epoch: CNN, mean pool 11, CNN, attention pool 6, small ResNet, mean pool 7, small ResNet, attention pool 10, small ResNet (no max pool), mean pool 7, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 14, ResNet-18, attention pool 15.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN, mean pool | 0.584 [0.529, 0.642] | -0.035 [-0.075, +0.002] | 0.582 | 0.533 | 0.713 | 0.573 | 0.463 |
| CNN, attention pool | 0.583 [0.525, 0.641] | -0.037 [-0.078, +0.005] | 0.537 | 0.475 | 0.765 | 0.567 | 0.475 |
| small ResNet, mean pool | 0.578 [0.524, 0.631] | -0.041 [-0.085, -0.003] | 0.558 | 0.525 | 0.740 | 0.510 | 0.463 |
| small ResNet, attention pool | 0.584 [0.529, 0.640] | -0.035 [-0.076, +0.003] | 0.582 | 0.525 | 0.765 | 0.541 | 0.463 |
| small ResNet (no max pool), mean pool | 0.584 [0.526, 0.641] | -0.036 [-0.079, +0.006] | 0.540 | 0.500 | 0.740 | 0.573 | 0.512 |
| small ResNet (no max pool), attention pool | 0.572 [0.515, 0.627] | -0.048 [-0.088, -0.009] | 0.540 | 0.508 | 0.738 | 0.541 | 0.475 |
| ResNet-18, mean pool | 0.503 [0.443, 0.563] | -0.116 [-0.171, -0.062] | 0.417 | 0.542 | 0.526 | 0.498 | 0.537 |
| ResNet-18, attention pool | 0.545 [0.487, 0.597] | -0.075 [-0.129, -0.022] | 0.454 | 0.483 | 0.630 | 0.504 | 0.650 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.597, 0.559, 0.590 | 0.582 ± 0.020 |
| CNN, attention pool | 0.606, 0.572, 0.601 | 0.593 ± 0.018 |
| small ResNet, mean pool | 0.583, 0.578, 0.555 | 0.572 ± 0.015 |
| small ResNet, attention pool | 0.556, 0.584, 0.565 | 0.568 ± 0.014 |
| small ResNet (no max pool), mean pool | 0.597, 0.556, 0.568 | 0.574 ± 0.021 |
| small ResNet (no max pool), attention pool | 0.595, 0.566, 0.599 | 0.587 ± 0.018 |
| ResNet-18, mean pool | 0.532, 0.504, 0.541 | 0.526 ± 0.019 |
| ResNet-18, attention pool | 0.553, 0.566, 0.510 | 0.543 ± 0.030 |

Median best epoch: CNN, mean pool 7, CNN, attention pool 9, small ResNet, mean pool 17, small ResNet, attention pool 7, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 9, ResNet-18, mean pool 13, ResNet-18, attention pool 11.
