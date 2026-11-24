# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN, mean pool | 0.591 [0.555, 0.628] | +0.055 [-0.014, +0.119] | 0.615 | 0.500 | 0.501 | 0.475 | 0.572 |
| CNN, attention pool | 0.559 [0.531, 0.587] | +0.023 [-0.038, +0.079] | 0.500 | 0.500 | 0.471 | 0.475 | 0.522 |
| CNN, mean ∥ max pool | 0.542 [0.513, 0.569] | +0.005 [-0.059, +0.065] | 0.463 | 0.478 | 0.486 | 0.487 | 0.562 |
| small ResNet, mean pool | 0.548 [0.517, 0.580] | +0.012 [-0.052, +0.076] | 0.522 | 0.500 | 0.471 | 0.463 | 0.497 |
| small ResNet, attention pool | 0.526 [0.502, 0.550] | -0.011 [-0.075, +0.052] | 0.543 | 0.500 | 0.500 | 0.463 | 0.392 |
| small ResNet, mean ∥ max pool | 0.572 [0.536, 0.607] | +0.035 [-0.033, +0.102] | 0.528 | 0.510 | 0.471 | 0.487 | 0.585 |
| small ResNet (no max pool), mean pool | 0.572 [0.537, 0.608] | +0.036 [-0.032, +0.104] | 0.525 | 0.500 | 0.486 | 0.487 | 0.620 |
| small ResNet (no max pool), attention pool | 0.531 [0.507, 0.555] | -0.006 [-0.069, +0.055] | 0.503 | 0.500 | 0.442 | 0.463 | 0.500 |
| ResNet-18, mean pool | 0.451 [0.406, 0.498] | -0.086 [-0.156, -0.014] | 0.426 | 0.462 | 0.517 | 0.400 | 0.331 |
| ResNet-18, attention pool | 0.518 [0.478, 0.558] | -0.018 [-0.091, +0.051] | 0.448 | 0.462 | 0.531 | 0.512 | 0.498 |
| CNN + ECA, mean pool | 0.573 [0.545, 0.601] | +0.037 [-0.029, +0.100] | 0.463 | 0.500 | 0.501 | 0.500 | 0.612 |
| CNN + ECA, attention pool | 0.580 [0.548, 0.608] | +0.043 [-0.024, +0.108] | 0.481 | 0.500 | 0.486 | 0.500 | 0.613 |
| CNN + CBAM, mean pool | 0.529 [0.493, 0.562] | -0.007 [-0.075, +0.057] | 0.466 | 0.397 | 0.471 | 0.487 | 0.520 |
| CNN + CBAM, attention pool | 0.565 [0.533, 0.601] | +0.028 [-0.036, +0.092] | 0.500 | 0.494 | 0.502 | 0.500 | 0.523 |
| small ResNet + ECA, mean pool | 0.554 [0.522, 0.587] | +0.018 [-0.050, +0.089] | 0.473 | 0.484 | 0.500 | 0.500 | 0.518 |
| small ResNet + ECA, attention pool | 0.533 [0.498, 0.566] | -0.003 [-0.070, +0.061] | 0.463 | 0.494 | 0.486 | 0.463 | 0.525 |
| small ResNet + CBAM, mean pool | 0.567 [0.539, 0.595] | +0.030 [-0.040, +0.094] | 0.522 | 0.500 | 0.500 | 0.463 | 0.539 |
| small ResNet + CBAM, attention pool | 0.517 [0.482, 0.553] | -0.019 [-0.091, +0.048] | 0.522 | 0.500 | 0.427 | 0.400 | 0.469 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.576, 0.546, 0.554 | 0.558 ± 0.016 |
| CNN, attention pool | 0.558, 0.531, 0.572 | 0.554 ± 0.021 |
| CNN, mean ∥ max pool | 0.478, 0.537, 0.550 | 0.522 ± 0.039 |
| small ResNet, mean pool | 0.563, 0.555, 0.531 | 0.550 ± 0.017 |
| small ResNet, attention pool | 0.535, 0.525, 0.539 | 0.533 ± 0.007 |
| small ResNet, mean ∥ max pool | 0.537, 0.584, 0.525 | 0.549 ± 0.031 |
| small ResNet (no max pool), mean pool | 0.564, 0.547, 0.564 | 0.558 ± 0.010 |
| small ResNet (no max pool), attention pool | 0.531, 0.556, 0.465 | 0.518 ± 0.047 |
| ResNet-18, mean pool | 0.473, 0.455, 0.457 | 0.462 ± 0.010 |
| ResNet-18, attention pool | 0.545, 0.530, 0.470 | 0.515 ± 0.040 |
| CNN + ECA, mean pool | 0.567, 0.557, 0.522 | 0.549 ± 0.024 |
| CNN + ECA, attention pool | 0.587, 0.574, 0.534 | 0.565 ± 0.027 |
| CNN + CBAM, mean pool | 0.581, 0.512, 0.573 | 0.555 ± 0.038 |
| CNN + CBAM, attention pool | 0.555, 0.542, 0.562 | 0.553 ± 0.010 |
| small ResNet + ECA, mean pool | 0.553, 0.561, 0.559 | 0.558 ± 0.004 |
| small ResNet + ECA, attention pool | 0.481, 0.591, 0.526 | 0.533 ± 0.055 |
| small ResNet + CBAM, mean pool | 0.563, 0.548, 0.594 | 0.568 ± 0.023 |
| small ResNet + CBAM, attention pool | 0.539, 0.547, 0.540 | 0.542 ± 0.004 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 7, CNN, mean ∥ max pool 7, small ResNet, mean pool 9, small ResNet, attention pool 9, small ResNet, mean ∥ max pool 4, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 9, ResNet-18, mean pool 7, ResNet-18, attention pool 10, CNN + ECA, mean pool 8, CNN + ECA, attention pool 7, CNN + CBAM, mean pool 8, CNN + CBAM, attention pool 8, small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 5, small ResNet + CBAM, mean pool 5, small ResNet + CBAM, attention pool 8.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.050 [-0.090, -0.007] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.023 [-0.013, +0.063] |
| CNN + ECA, mean pool | CNN, mean pool | -0.018 [-0.055, +0.021] |
| CNN + ECA, attention pool | CNN, attention pool | +0.021 [-0.020, +0.064] |
| CNN + CBAM, mean pool | CNN, mean pool | -0.062 [-0.107, -0.019] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.006 [-0.030, +0.046] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.006 [-0.030, +0.043] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.008 [-0.020, +0.039] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.019 [-0.017, +0.054] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.008 [-0.053, +0.035] |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN, mean pool | 0.523 [0.475, 0.570] | -0.082 [-0.143, -0.020] | 0.514 | 0.575 | 0.627 | 0.494 | 0.500 |
| CNN, attention pool | 0.519 [0.472, 0.566] | -0.086 [-0.152, -0.023] | 0.537 | 0.600 | 0.668 | 0.500 | 0.500 |
| CNN, mean ∥ max pool | 0.560 [0.509, 0.606] | -0.045 [-0.106, +0.010] | 0.551 | 0.583 | 0.500 | 0.526 | 0.525 |
| small ResNet, mean pool | 0.516 [0.465, 0.564] | -0.089 [-0.152, -0.026] | 0.503 | 0.583 | 0.339 | 0.526 | 0.475 |
| small ResNet, attention pool | 0.545 [0.498, 0.591] | -0.060 [-0.121, -0.001] | 0.558 | 0.525 | 0.794 | 0.500 | 0.512 |
| small ResNet, mean ∥ max pool | 0.556 [0.503, 0.604] | -0.049 [-0.110, +0.010] | 0.522 | 0.575 | 0.722 | 0.456 | 0.450 |
| small ResNet (no max pool), mean pool | 0.568 [0.515, 0.622] | -0.037 [-0.102, +0.028] | 0.559 | 0.600 | 0.715 | 0.510 | 0.525 |
| small ResNet (no max pool), attention pool | 0.570 [0.517, 0.623] | -0.035 [-0.101, +0.030] | 0.508 | 0.508 | 0.734 | 0.510 | 0.575 |
| ResNet-18, mean pool | 0.538 [0.485, 0.588] | -0.067 [-0.139, +0.000] | 0.569 | 0.583 | 0.463 | 0.510 | 0.537 |
| ResNet-18, attention pool | 0.497 [0.444, 0.550] | -0.108 [-0.180, -0.037] | 0.490 | 0.575 | 0.404 | 0.494 | 0.575 |
| CNN + ECA, mean pool | 0.536 [0.491, 0.580] | -0.069 [-0.128, -0.009] | 0.505 | 0.550 | 0.709 | 0.510 | 0.500 |
| CNN + ECA, attention pool | 0.578 [0.523, 0.632] | -0.027 [-0.088, +0.030] | 0.518 | 0.517 | 0.731 | 0.537 | 0.613 |
| CNN + CBAM, mean pool | 0.594 [0.541, 0.646] | -0.011 [-0.068, +0.043] | 0.510 | 0.525 | 0.806 | 0.526 | 0.537 |
| CNN + CBAM, attention pool | 0.566 [0.514, 0.619] | -0.039 [-0.101, +0.021] | 0.474 | 0.500 | 0.725 | 0.494 | 0.562 |
| small ResNet + ECA, mean pool | 0.577 [0.526, 0.627] | -0.028 [-0.093, +0.036] | 0.505 | 0.542 | 0.627 | 0.510 | 0.587 |
| small ResNet + ECA, attention pool | 0.608 [0.556, 0.659] | +0.003 [-0.058, +0.060] | 0.603 | 0.550 | 0.731 | 0.526 | 0.625 |
| small ResNet + CBAM, mean pool | 0.603 [0.551, 0.653] | -0.002 [-0.058, +0.053] | 0.550 | 0.575 | 0.765 | 0.510 | 0.562 |
| small ResNet + CBAM, attention pool | 0.558 [0.507, 0.608] | -0.047 [-0.107, +0.010] | 0.494 | 0.533 | 0.538 | 0.526 | 0.575 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.518, 0.563, 0.543 | 0.541 ± 0.022 |
| CNN, attention pool | 0.498, 0.504, 0.541 | 0.514 ± 0.024 |
| CNN, mean ∥ max pool | 0.561, 0.518, 0.556 | 0.545 ± 0.023 |
| small ResNet, mean pool | 0.572, 0.553, 0.533 | 0.553 ± 0.020 |
| small ResNet, attention pool | 0.564, 0.544, 0.556 | 0.555 ± 0.010 |
| small ResNet, mean ∥ max pool | 0.530, 0.537, 0.584 | 0.551 ± 0.030 |
| small ResNet (no max pool), mean pool | 0.553, 0.530, 0.536 | 0.540 ± 0.012 |
| small ResNet (no max pool), attention pool | 0.539, 0.530, 0.577 | 0.549 ± 0.025 |
| ResNet-18, mean pool | 0.534, 0.505, 0.530 | 0.523 ± 0.016 |
| ResNet-18, attention pool | 0.491, 0.499, 0.495 | 0.495 ± 0.004 |
| CNN + ECA, mean pool | 0.519, 0.483, 0.540 | 0.514 ± 0.028 |
| CNN + ECA, attention pool | 0.539, 0.529, 0.598 | 0.555 ± 0.037 |
| CNN + CBAM, mean pool | 0.534, 0.543, 0.595 | 0.558 ± 0.033 |
| CNN + CBAM, attention pool | 0.549, 0.533, 0.565 | 0.549 ± 0.016 |
| small ResNet + ECA, mean pool | 0.536, 0.565, 0.597 | 0.566 ± 0.030 |
| small ResNet + ECA, attention pool | 0.545, 0.543, 0.541 | 0.543 ± 0.002 |
| small ResNet + CBAM, mean pool | 0.586, 0.525, 0.614 | 0.575 ± 0.045 |
| small ResNet + CBAM, attention pool | 0.506, 0.572, 0.561 | 0.546 ± 0.035 |

Median best epoch: CNN, mean pool 5, CNN, attention pool 6, CNN, mean ∥ max pool 6, small ResNet, mean pool 7, small ResNet, attention pool 6, small ResNet, mean ∥ max pool 9, small ResNet (no max pool), mean pool 14, small ResNet (no max pool), attention pool 12, ResNet-18, mean pool 8, ResNet-18, attention pool 8, CNN + ECA, mean pool 8, CNN + ECA, attention pool 9, CNN + CBAM, mean pool 7, CNN + CBAM, attention pool 7, small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 4, small ResNet + CBAM, mean pool 6, small ResNet + CBAM, attention pool 4.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.037 [-0.005, +0.079] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.039 [-0.007, +0.086] |
| CNN + ECA, mean pool | CNN, mean pool | +0.013 [-0.026, +0.054] |
| CNN + ECA, attention pool | CNN, attention pool | +0.059 [+0.015, +0.104] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.070 [+0.022, +0.120] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.048 [+0.006, +0.089] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.061 [+0.016, +0.106] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.063 [+0.024, +0.103] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.087 [+0.038, +0.132] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.014 [-0.026, +0.055] |
