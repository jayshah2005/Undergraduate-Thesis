# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| CNN, mean pool | 0.474 [0.419, 0.525] | -0.082 [-0.173, +0.002] | 0.600 | 0.466 | 0.511 | 0.488 | 0.760 |
| CNN, attention pool | 0.447 [0.387, 0.510] | -0.110 [-0.182, -0.035] | 0.478 | 0.447 | 0.543 | 0.488 | 0.750 |
| CNN, mean ∥ max pool | 0.435 [0.357, 0.514] | -0.121 [-0.213, -0.032] | 0.543 | 0.394 | 0.543 | 0.562 | 0.510 |
| small ResNet, mean pool | 0.513 [0.441, 0.589] | -0.043 [-0.121, +0.031] | 0.568 | 0.375 | 0.628 | 0.491 | 0.490 |
| small ResNet, attention pool | 0.432 [0.377, 0.488] | -0.125 [-0.213, -0.038] | 0.472 | 0.386 | 0.521 | 0.488 | 0.740 |
| small ResNet, mean ∥ max pool | 0.459 [0.405, 0.522] | -0.097 [-0.188, -0.002] | 0.445 | 0.500 | 0.255 | 0.488 | 0.719 |
| small ResNet (no max pool), mean pool | 0.448 [0.390, 0.501] | -0.109 [-0.203, -0.025] | 0.611 | 0.409 | 0.670 | 0.500 | 0.510 |
| small ResNet (no max pool), attention pool | 0.488 [0.437, 0.536] | -0.068 [-0.135, -0.001] | 0.558 | 0.458 | 0.681 | 0.500 | 0.562 |
| ResNet-18, mean pool | 0.440 [0.365, 0.518] | -0.117 [-0.219, -0.014] | 0.575 | 0.538 | 0.585 | 0.491 | 0.135 |
| ResNet-18, attention pool | 0.440 [0.368, 0.517] | -0.117 [-0.216, -0.019] | 0.470 | 0.504 | 0.532 | 0.539 | 0.594 |
| CNN + ECA, mean pool | 0.453 [0.388, 0.518] | -0.103 [-0.196, -0.014] | 0.611 | 0.504 | 0.606 | 0.500 | 0.500 |
| CNN + ECA, attention pool | 0.517 [0.455, 0.579] | -0.039 [-0.125, +0.038] | 0.718 | 0.527 | 0.500 | 0.500 | 0.760 |
| CNN + CBAM, mean pool | 0.425 [0.365, 0.490] | -0.131 [-0.240, -0.023] | 0.550 | 0.508 | 0.553 | 0.488 | 0.521 |
| CNN + CBAM, attention pool | 0.468 [0.412, 0.532] | -0.089 [-0.189, +0.012] | 0.457 | 0.538 | 0.521 | 0.488 | 0.500 |
| small ResNet + ECA, mean pool | 0.419 [0.348, 0.488] | -0.138 [-0.237, -0.042] | 0.390 | 0.485 | 0.660 | 0.488 | 0.500 |
| small ResNet + ECA, attention pool | 0.499 [0.453, 0.552] | -0.058 [-0.142, +0.028] | 0.500 | 0.371 | 0.649 | 0.500 | 0.479 |
| small ResNet + CBAM, mean pool | 0.451 [0.387, 0.507] | -0.105 [-0.190, -0.022] | 0.508 | 0.564 | 0.543 | 0.500 | 0.510 |
| small ResNet + CBAM, attention pool | 0.453 [0.406, 0.507] | -0.103 [-0.185, -0.029] | 0.500 | 0.473 | 0.511 | 0.562 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.479, 0.424, 0.397 | 0.434 ± 0.042 |
| CNN, attention pool | 0.451, 0.382, 0.486 | 0.439 ± 0.053 |
| CNN, mean ∥ max pool | 0.412, 0.431, 0.438 | 0.427 ± 0.013 |
| small ResNet, mean pool | 0.507, 0.580, 0.526 | 0.538 ± 0.038 |
| small ResNet, attention pool | 0.458, 0.443, 0.431 | 0.444 ± 0.013 |
| small ResNet, mean ∥ max pool | 0.393, 0.438, 0.573 | 0.468 ± 0.093 |
| small ResNet (no max pool), mean pool | 0.463, 0.445, 0.389 | 0.432 ± 0.038 |
| small ResNet (no max pool), attention pool | 0.458, 0.474, 0.472 | 0.468 ± 0.009 |
| ResNet-18, mean pool | 0.404, 0.414, 0.478 | 0.432 ± 0.040 |
| ResNet-18, attention pool | 0.430, 0.409, 0.435 | 0.425 ± 0.014 |
| CNN + ECA, mean pool | 0.438, 0.463, 0.443 | 0.448 ± 0.014 |
| CNN + ECA, attention pool | 0.546, 0.425, 0.424 | 0.465 ± 0.070 |
| CNN + CBAM, mean pool | 0.472, 0.419, 0.428 | 0.440 ± 0.028 |
| CNN + CBAM, attention pool | 0.510, 0.519, 0.496 | 0.508 ± 0.011 |
| small ResNet + ECA, mean pool | 0.411, 0.449, 0.496 | 0.452 ± 0.043 |
| small ResNet + ECA, attention pool | 0.502, 0.456, 0.470 | 0.476 ± 0.023 |
| small ResNet + CBAM, mean pool | 0.463, 0.483, 0.518 | 0.488 ± 0.028 |
| small ResNet + CBAM, attention pool | 0.428, 0.458, 0.434 | 0.440 ± 0.016 |

Median best epoch: CNN, mean pool 7, CNN, attention pool 10, CNN, mean ∥ max pool 2, small ResNet, mean pool 14, small ResNet, attention pool 2, small ResNet, mean ∥ max pool 1, small ResNet (no max pool), mean pool 13, small ResNet (no max pool), attention pool 4, ResNet-18, mean pool 6, ResNet-18, attention pool 7, CNN + ECA, mean pool 11, CNN + ECA, attention pool 7, CNN + CBAM, mean pool 14, CNN + CBAM, attention pool 6, small ResNet + ECA, mean pool 11, small ResNet + ECA, attention pool 5, small ResNet + CBAM, mean pool 5, small ResNet + CBAM, attention pool 2.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.039 [-0.107, +0.042] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.054 [-0.159, +0.056] |
| CNN + ECA, mean pool | CNN, mean pool | -0.021 [-0.088, +0.056] |
| CNN + ECA, attention pool | CNN, attention pool | +0.071 [-0.003, +0.145] |
| CNN + CBAM, mean pool | CNN, mean pool | -0.049 [-0.131, +0.039] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.021 [-0.056, +0.114] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | -0.094 [-0.204, +0.009] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.067 [-0.004, +0.144] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | -0.062 [-0.149, +0.025] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.021 [-0.052, +0.096] |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| CNN, mean pool | 0.557 [0.498, 0.612] | +0.009 [-0.071, +0.090] | 0.640 | 0.589 | 0.542 | 0.525 | 0.582 |
| CNN, attention pool | 0.492 [0.442, 0.541] | -0.055 [-0.127, +0.016] | 0.553 | 0.521 | 0.481 | 0.525 | 0.555 |
| CNN, mean ∥ max pool | 0.534 [0.486, 0.578] | -0.013 [-0.079, +0.050] | 0.682 | 0.583 | 0.500 | 0.560 | 0.542 |
| small ResNet, mean pool | 0.503 [0.450, 0.556] | -0.044 [-0.119, +0.029] | 0.612 | 0.564 | 0.505 | 0.450 | 0.512 |
| small ResNet, attention pool | 0.509 [0.456, 0.560] | -0.038 [-0.115, +0.037] | 0.695 | 0.540 | 0.505 | 0.484 | 0.481 |
| small ResNet, mean ∥ max pool | 0.523 [0.466, 0.576] | -0.024 [-0.103, +0.053] | 0.570 | 0.556 | 0.596 | 0.512 | 0.480 |
| small ResNet (no max pool), mean pool | 0.509 [0.453, 0.562] | -0.038 [-0.109, +0.035] | 0.586 | 0.620 | 0.505 | 0.467 | 0.498 |
| small ResNet (no max pool), attention pool | 0.519 [0.467, 0.570] | -0.028 [-0.098, +0.044] | 0.671 | 0.604 | 0.469 | 0.467 | 0.541 |
| ResNet-18, mean pool | 0.533 [0.479, 0.586] | -0.015 [-0.095, +0.065] | 0.487 | 0.592 | 0.518 | 0.580 | 0.539 |
| ResNet-18, attention pool | 0.510 [0.462, 0.559] | -0.038 [-0.117, +0.043] | 0.590 | 0.500 | 0.518 | 0.575 | 0.586 |
| CNN + ECA, mean pool | 0.502 [0.456, 0.547] | -0.046 [-0.118, +0.027] | 0.632 | 0.528 | 0.547 | 0.439 | 0.498 |
| CNN + ECA, attention pool | 0.510 [0.469, 0.552] | -0.037 [-0.104, +0.032] | 0.684 | 0.500 | 0.469 | 0.498 | 0.571 |
| CNN + CBAM, mean pool | 0.562 [0.514, 0.608] | +0.014 [-0.058, +0.084] | 0.695 | 0.500 | 0.621 | 0.511 | 0.538 |
| CNN + CBAM, attention pool | 0.476 [0.433, 0.517] | -0.071 [-0.139, -0.002] | 0.526 | 0.500 | 0.505 | 0.439 | 0.585 |
| small ResNet + ECA, mean pool | 0.547 [0.496, 0.596] | -0.001 [-0.073, +0.071] | 0.697 | 0.568 | 0.547 | 0.498 | 0.526 |
| small ResNet + ECA, attention pool | 0.485 [0.438, 0.530] | -0.062 [-0.129, +0.006] | 0.645 | 0.500 | 0.481 | 0.484 | 0.493 |
| small ResNet + CBAM, mean pool | 0.521 [0.480, 0.562] | -0.026 [-0.090, +0.035] | 0.711 | 0.556 | 0.500 | 0.543 | 0.513 |
| small ResNet + CBAM, attention pool | 0.509 [0.469, 0.547] | -0.038 [-0.103, +0.025] | 0.618 | 0.500 | 0.512 | 0.567 | 0.555 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.524, 0.533, 0.598 | 0.552 ± 0.040 |
| CNN, attention pool | 0.525, 0.454, 0.496 | 0.492 ± 0.036 |
| CNN, mean ∥ max pool | 0.555, 0.484, 0.498 | 0.512 ± 0.038 |
| small ResNet, mean pool | 0.498, 0.565, 0.495 | 0.519 ± 0.040 |
| small ResNet, attention pool | 0.505, 0.513, 0.526 | 0.514 ± 0.010 |
| small ResNet, mean ∥ max pool | 0.490, 0.602, 0.552 | 0.548 ± 0.056 |
| small ResNet (no max pool), mean pool | 0.514, 0.528, 0.522 | 0.521 ± 0.007 |
| small ResNet (no max pool), attention pool | 0.539, 0.504, 0.465 | 0.503 ± 0.037 |
| ResNet-18, mean pool | 0.547, 0.471, 0.495 | 0.504 ± 0.038 |
| ResNet-18, attention pool | 0.539, 0.488, 0.496 | 0.508 ± 0.027 |
| CNN + ECA, mean pool | 0.474, 0.517, 0.608 | 0.533 ± 0.068 |
| CNN + ECA, attention pool | 0.517, 0.492, 0.534 | 0.514 ± 0.021 |
| CNN + CBAM, mean pool | 0.519, 0.537, 0.565 | 0.540 ± 0.024 |
| CNN + CBAM, attention pool | 0.486, 0.488, 0.511 | 0.495 ± 0.014 |
| small ResNet + ECA, mean pool | 0.495, 0.578, 0.548 | 0.540 ± 0.042 |
| small ResNet + ECA, attention pool | 0.497, 0.520, 0.474 | 0.497 ± 0.023 |
| small ResNet + CBAM, mean pool | 0.492, 0.524, 0.498 | 0.505 ± 0.017 |
| small ResNet + CBAM, attention pool | 0.502, 0.503, 0.510 | 0.505 ± 0.004 |

Median best epoch: CNN, mean pool 10, CNN, attention pool 5, CNN, mean ∥ max pool 8, small ResNet, mean pool 8, small ResNet, attention pool 9, small ResNet, mean ∥ max pool 12, small ResNet (no max pool), mean pool 4, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 9, ResNet-18, attention pool 10, CNN + ECA, mean pool 8, CNN + ECA, attention pool 5, CNN + CBAM, mean pool 10, CNN + CBAM, attention pool 5, small ResNet + ECA, mean pool 5, small ResNet + ECA, attention pool 8, small ResNet + CBAM, mean pool 9, small ResNet + CBAM, attention pool 7.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.022 [-0.084, +0.038] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.020 [-0.029, +0.068] |
| CNN + ECA, mean pool | CNN, mean pool | -0.055 [-0.108, -0.001] |
| CNN + ECA, attention pool | CNN, attention pool | +0.018 [-0.026, +0.058] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.005 [-0.050, +0.058] |
| CNN + CBAM, attention pool | CNN, attention pool | -0.016 [-0.066, +0.033] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.043 [-0.004, +0.091] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.024 [-0.061, +0.014] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.018 [-0.030, +0.066] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.000 [-0.045, +0.045] |
