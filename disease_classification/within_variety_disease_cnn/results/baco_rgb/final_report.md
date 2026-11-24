# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN, mean pool | 0.370 [0.344, 0.398] | -0.049 [-0.097, +0.002] | 0.500 | 0.519 | 0.304 | 0.400 | 0.517 |
| CNN, attention pool | 0.440 [0.407, 0.472] | +0.022 [-0.028, +0.074] | 0.500 | 0.500 | 0.357 | 0.425 | 0.543 |
| CNN, mean ∥ max pool | 0.507 [0.464, 0.548] | +0.088 [+0.037, +0.142] | 0.628 | 0.500 | 0.514 | 0.400 | 0.528 |
| small ResNet, mean pool | 0.535 [0.511, 0.559] | +0.117 [+0.065, +0.170] | 0.466 | 0.500 | 0.454 | 0.525 | 0.500 |
| small ResNet, attention pool | 0.489 [0.453, 0.524] | +0.071 [+0.027, +0.115] | 0.500 | 0.380 | 0.500 | 0.400 | 0.500 |
| small ResNet, mean ∥ max pool | 0.481 [0.438, 0.525] | +0.063 [+0.008, +0.116] | 0.551 | 0.515 | 0.482 | 0.412 | 0.500 |
| small ResNet (no max pool), mean pool | 0.497 [0.459, 0.532] | +0.079 [+0.025, +0.131] | 0.417 | 0.419 | 0.500 | 0.500 | 0.500 |
| small ResNet (no max pool), attention pool | 0.534 [0.505, 0.561] | +0.115 [+0.065, +0.166] | 0.500 | 0.515 | 0.500 | 0.500 | 0.500 |
| ResNet-18, mean pool | 0.554 [0.505, 0.600] | +0.135 [+0.064, +0.205] | 0.514 | 0.578 | 0.500 | 0.600 | 0.402 |
| ResNet-18, attention pool | 0.582 [0.543, 0.619] | +0.163 [+0.102, +0.224] | 0.535 | 0.565 | 0.500 | 0.475 | 0.450 |
| CNN + ECA, mean pool | 0.486 [0.459, 0.515] | +0.068 [+0.015, +0.120] | 0.500 | 0.519 | 0.532 | 0.450 | 0.500 |
| CNN + ECA, attention pool | 0.489 [0.458, 0.519] | +0.070 [+0.025, +0.115] | 0.500 | 0.515 | 0.464 | 0.400 | 0.500 |
| CNN + CBAM, mean pool | 0.511 [0.466, 0.558] | +0.093 [+0.032, +0.156] | 0.537 | 0.509 | 0.382 | 0.450 | 0.556 |
| CNN + CBAM, attention pool | 0.557 [0.519, 0.594] | +0.139 [+0.075, +0.205] | 0.500 | 0.486 | 0.486 | 0.475 | 0.476 |
| small ResNet + ECA, mean pool | 0.490 [0.444, 0.535] | +0.072 [+0.009, +0.136] | 0.479 | 0.512 | 0.489 | 0.575 | 0.500 |
| small ResNet + ECA, attention pool | 0.454 [0.409, 0.500] | +0.036 [-0.014, +0.088] | 0.415 | 0.546 | 0.500 | 0.375 | 0.500 |
| small ResNet + CBAM, mean pool | 0.580 [0.537, 0.622] | +0.162 [+0.110, +0.212] | 0.641 | 0.506 | 0.500 | 0.513 | 0.567 |
| small ResNet + CBAM, attention pool | 0.484 [0.426, 0.540] | +0.066 [+0.008, +0.121] | 0.443 | 0.505 | 0.568 | 0.388 | 0.504 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.405, 0.434, 0.365 | 0.401 ± 0.035 |
| CNN, attention pool | 0.509, 0.369, 0.440 | 0.439 ± 0.070 |
| CNN, mean ∥ max pool | 0.510, 0.419, 0.526 | 0.485 ± 0.058 |
| small ResNet, mean pool | 0.557, 0.488, 0.510 | 0.518 ± 0.036 |
| small ResNet, attention pool | 0.531, 0.381, 0.449 | 0.454 ± 0.075 |
| small ResNet, mean ∥ max pool | 0.484, 0.466, 0.521 | 0.490 ± 0.028 |
| small ResNet (no max pool), mean pool | 0.523, 0.470, 0.432 | 0.475 ± 0.046 |
| small ResNet (no max pool), attention pool | 0.585, 0.481, 0.412 | 0.492 ± 0.087 |
| ResNet-18, mean pool | 0.515, 0.516, 0.497 | 0.509 ± 0.011 |
| ResNet-18, attention pool | 0.576, 0.535, 0.539 | 0.550 ± 0.023 |
| CNN + ECA, mean pool | 0.392, 0.486, 0.499 | 0.459 ± 0.058 |
| CNN + ECA, attention pool | 0.510, 0.443, 0.403 | 0.452 ± 0.054 |
| CNN + CBAM, mean pool | 0.421, 0.546, 0.585 | 0.517 ± 0.085 |
| CNN + CBAM, attention pool | 0.419, 0.598, 0.574 | 0.530 ± 0.097 |
| small ResNet + ECA, mean pool | 0.538, 0.424, 0.449 | 0.470 ± 0.060 |
| small ResNet + ECA, attention pool | 0.487, 0.426, 0.526 | 0.480 ± 0.050 |
| small ResNet + CBAM, mean pool | 0.552, 0.513, 0.459 | 0.508 ± 0.047 |
| small ResNet + CBAM, attention pool | 0.411, 0.514, 0.514 | 0.480 ± 0.059 |

Median best epoch: CNN, mean pool 3, CNN, attention pool 2, CNN, mean ∥ max pool 5, small ResNet, mean pool 2, small ResNet, attention pool 3, small ResNet, mean ∥ max pool 4, small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 2, ResNet-18, mean pool 7, ResNet-18, attention pool 14, CNN + ECA, mean pool 3, CNN + ECA, attention pool 1, CNN + CBAM, mean pool 5, CNN + CBAM, attention pool 2, small ResNet + ECA, mean pool 3, small ResNet + ECA, attention pool 3, small ResNet + CBAM, mean pool 3, small ResNet + CBAM, attention pool 6.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.137 [+0.092, +0.180] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.054 [-0.100, -0.007] |
| CNN + ECA, mean pool | CNN, mean pool | +0.116 [+0.087, +0.145] |
| CNN + ECA, attention pool | CNN, attention pool | +0.048 [+0.008, +0.087] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.142 [+0.099, +0.186] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.117 [+0.070, +0.166] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | -0.045 [-0.096, +0.008] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.035 [-0.078, +0.006] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.045 [-0.002, +0.093] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.005 [-0.060, +0.049] |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN, mean pool | 0.525 [0.475, 0.576] | -0.018 [-0.077, +0.037] | 0.649 | 0.524 | 0.502 | 0.500 | 0.397 |
| CNN, attention pool | 0.548 [0.500, 0.595] | +0.005 [-0.060, +0.071] | 0.646 | 0.513 | 0.554 | 0.500 | 0.449 |
| CNN, mean ∥ max pool | 0.516 [0.466, 0.567] | -0.027 [-0.078, +0.021] | 0.427 | 0.568 | 0.509 | 0.600 | 0.404 |
| small ResNet, mean pool | 0.542 [0.497, 0.585] | -0.001 [-0.049, +0.047] | 0.528 | 0.552 | 0.554 | 0.600 | 0.383 |
| small ResNet, attention pool | 0.573 [0.533, 0.612] | +0.030 [-0.024, +0.080] | 0.500 | 0.596 | 0.531 | 0.600 | 0.463 |
| small ResNet, mean ∥ max pool | 0.491 [0.439, 0.544] | -0.052 [-0.114, +0.013] | 0.514 | 0.438 | 0.484 | 0.540 | 0.429 |
| small ResNet (no max pool), mean pool | 0.563 [0.521, 0.607] | +0.020 [-0.038, +0.081] | 0.500 | 0.515 | 0.509 | 0.600 | 0.566 |
| small ResNet (no max pool), attention pool | 0.534 [0.492, 0.576] | -0.009 [-0.054, +0.034] | 0.500 | 0.570 | 0.509 | 0.580 | 0.378 |
| ResNet-18, mean pool | 0.515 [0.463, 0.567] | -0.028 [-0.089, +0.032] | 0.559 | 0.494 | 0.502 | 0.500 | 0.486 |
| ResNet-18, attention pool | 0.563 [0.512, 0.615] | +0.020 [-0.037, +0.075] | 0.712 | 0.479 | 0.456 | 0.600 | 0.526 |
| CNN + ECA, mean pool | 0.530 [0.478, 0.580] | -0.013 [-0.060, +0.033] | 0.569 | 0.524 | 0.509 | 0.580 | 0.396 |
| CNN + ECA, attention pool | 0.528 [0.479, 0.578] | -0.015 [-0.068, +0.037] | 0.573 | 0.534 | 0.460 | 0.600 | 0.401 |
| CNN + CBAM, mean pool | 0.566 [0.517, 0.615] | +0.023 [-0.029, +0.072] | 0.521 | 0.542 | 0.521 | 0.580 | 0.524 |
| CNN + CBAM, attention pool | 0.539 [0.490, 0.589] | -0.005 [-0.053, +0.048] | 0.450 | 0.560 | 0.531 | 0.620 | 0.412 |
| small ResNet + ECA, mean pool | 0.570 [0.527, 0.611] | +0.027 [-0.031, +0.084] | 0.641 | 0.591 | 0.554 | 0.500 | 0.444 |
| small ResNet + ECA, attention pool | 0.554 [0.508, 0.600] | +0.011 [-0.033, +0.055] | 0.500 | 0.614 | 0.486 | 0.600 | 0.441 |
| small ResNet + CBAM, mean pool | 0.530 [0.482, 0.573] | -0.014 [-0.069, +0.040] | 0.545 | 0.623 | 0.509 | 0.520 | 0.359 |
| small ResNet + CBAM, attention pool | 0.554 [0.506, 0.601] | +0.011 [-0.040, +0.059] | 0.552 | 0.614 | 0.509 | 0.580 | 0.425 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.508, 0.548, 0.524 | 0.527 ± 0.020 |
| CNN, attention pool | 0.543, 0.559, 0.544 | 0.549 ± 0.009 |
| CNN, mean ∥ max pool | 0.528, 0.523, 0.517 | 0.523 ± 0.006 |
| small ResNet, mean pool | 0.535, 0.529, 0.526 | 0.530 ± 0.005 |
| small ResNet, attention pool | 0.547, 0.572, 0.522 | 0.547 ± 0.025 |
| small ResNet, mean ∥ max pool | 0.501, 0.499, 0.524 | 0.508 ± 0.014 |
| small ResNet (no max pool), mean pool | 0.577, 0.500, 0.552 | 0.543 ± 0.039 |
| small ResNet (no max pool), attention pool | 0.520, 0.549, 0.514 | 0.528 ± 0.019 |
| ResNet-18, mean pool | 0.476, 0.510, 0.538 | 0.508 ± 0.031 |
| ResNet-18, attention pool | 0.553, 0.549, 0.559 | 0.553 ± 0.005 |
| CNN + ECA, mean pool | 0.516, 0.572, 0.535 | 0.541 ± 0.028 |
| CNN + ECA, attention pool | 0.537, 0.533, 0.555 | 0.541 ± 0.012 |
| CNN + CBAM, mean pool | 0.513, 0.531, 0.544 | 0.529 ± 0.016 |
| CNN + CBAM, attention pool | 0.539, 0.542, 0.546 | 0.543 ± 0.004 |
| small ResNet + ECA, mean pool | 0.520, 0.561, 0.525 | 0.535 ± 0.023 |
| small ResNet + ECA, attention pool | 0.534, 0.527, 0.546 | 0.536 ± 0.009 |
| small ResNet + CBAM, mean pool | 0.507, 0.522, 0.575 | 0.535 ± 0.035 |
| small ResNet + CBAM, attention pool | 0.531, 0.540, 0.580 | 0.550 ± 0.026 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5, CNN, mean ∥ max pool 4, small ResNet, mean pool 7, small ResNet, attention pool 6, small ResNet, mean ∥ max pool 4, small ResNet (no max pool), mean pool 7, small ResNet (no max pool), attention pool 7, ResNet-18, mean pool 12, ResNet-18, attention pool 8, CNN + ECA, mean pool 10, CNN + ECA, attention pool 8, CNN + CBAM, mean pool 11, CNN + CBAM, attention pool 9, small ResNet + ECA, mean pool 5, small ResNet + ECA, attention pool 6, small ResNet + CBAM, mean pool 6, small ResNet + CBAM, attention pool 7.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.009 [-0.056, +0.041] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.051 [-0.108, +0.005] |
| CNN + ECA, mean pool | CNN, mean pool | +0.005 [-0.039, +0.052] |
| CNN + ECA, attention pool | CNN, attention pool | -0.020 [-0.073, +0.032] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.041 [-0.008, +0.091] |
| CNN + CBAM, attention pool | CNN, attention pool | -0.009 [-0.065, +0.049] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.028 [-0.022, +0.076] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.019 [-0.057, +0.018] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | -0.013 [-0.053, +0.026] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.019 [-0.057, +0.019] |
