# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| CNN, mean pool | 0.415 [0.337, 0.494] | -0.024 [-0.125, +0.087] | 0.475 | 0.436 | 0.500 | 0.351 | 0.458 |
| CNN, attention pool | 0.410 [0.367, 0.461] | -0.029 [-0.129, +0.074] | 0.500 | 0.477 | 0.500 | 0.363 | 0.510 |
| CNN, mean ∥ max pool | 0.420 [0.389, 0.463] | -0.020 [-0.108, +0.074] | 0.500 | 0.538 | 0.500 | 0.393 | 0.531 |
| small ResNet, mean pool | 0.368 [0.323, 0.400] | -0.071 [-0.149, +0.009] | 0.487 | 0.375 | 0.500 | 0.381 | 0.500 |
| small ResNet, attention pool | 0.392 [0.347, 0.421] | -0.047 [-0.127, +0.037] | 0.475 | 0.386 | 0.500 | 0.500 | 0.510 |
| small ResNet, mean ∥ max pool | 0.467 [0.432, 0.513] | +0.027 [-0.059, +0.113] | 0.500 | 0.455 | 0.500 | 0.467 | 0.740 |
| small ResNet (no max pool), mean pool | 0.431 [0.359, 0.507] | -0.008 [-0.112, +0.098] | 0.487 | 0.508 | 0.500 | 0.458 | 0.510 |
| small ResNet (no max pool), attention pool | 0.411 [0.400, 0.424] | -0.028 [-0.109, +0.060] | 0.500 | 0.500 | 0.500 | 0.500 | 0.125 |
| ResNet-18, mean pool | 0.395 [0.339, 0.447] | -0.044 [-0.127, +0.033] | 0.458 | 0.398 | 0.564 | 0.500 | 0.542 |
| ResNet-18, attention pool | 0.378 [0.331, 0.430] | -0.061 [-0.163, +0.041] | 0.399 | 0.477 | 0.521 | 0.452 | 0.552 |
| CNN + ECA, mean pool | 0.458 [0.402, 0.522] | +0.018 [-0.083, +0.126] | 0.500 | 0.413 | 0.500 | 0.315 | 0.500 |
| CNN + ECA, attention pool | 0.395 [0.319, 0.472] | -0.044 [-0.137, +0.056] | 0.462 | 0.413 | 0.500 | 0.435 | 0.510 |
| CNN + CBAM, mean pool | 0.445 [0.398, 0.480] | +0.006 [-0.097, +0.108] | 0.487 | 0.477 | 0.564 | 0.405 | 0.365 |
| CNN + CBAM, attention pool | 0.550 [0.507, 0.604] | +0.111 [+0.029, +0.192] | 0.500 | 0.455 | 0.500 | 0.530 | 0.500 |
| small ResNet + ECA, mean pool | 0.464 [0.409, 0.529] | +0.025 [-0.080, +0.133] | 0.500 | 0.458 | 0.500 | 0.351 | 0.458 |
| small ResNet + ECA, attention pool | 0.389 [0.321, 0.460] | -0.050 [-0.139, +0.045] | 0.462 | 0.500 | 0.500 | 0.411 | 0.115 |
| small ResNet + CBAM, mean pool | 0.457 [0.406, 0.500] | +0.018 [-0.056, +0.094] | 0.450 | 0.432 | 0.511 | 0.488 | 0.500 |
| small ResNet + CBAM, attention pool | 0.480 [0.412, 0.547] | +0.041 [-0.031, +0.112] | 0.450 | 0.409 | 0.500 | 0.565 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.442, 0.488, 0.423 | 0.451 ± 0.033 |
| CNN, attention pool | 0.415, 0.365, 0.417 | 0.399 ± 0.029 |
| CNN, mean ∥ max pool | 0.453, 0.424, 0.425 | 0.434 ± 0.017 |
| small ResNet, mean pool | 0.416, 0.422, 0.404 | 0.414 ± 0.009 |
| small ResNet, attention pool | 0.383, 0.399, 0.376 | 0.386 ± 0.012 |
| small ResNet, mean ∥ max pool | 0.464, 0.404, 0.418 | 0.428 ± 0.031 |
| small ResNet (no max pool), mean pool | 0.433, 0.392, 0.457 | 0.427 ± 0.033 |
| small ResNet (no max pool), attention pool | 0.409, 0.469, 0.432 | 0.437 ± 0.030 |
| ResNet-18, mean pool | 0.405, 0.334, 0.462 | 0.400 ± 0.064 |
| ResNet-18, attention pool | 0.418, 0.302, 0.442 | 0.387 ± 0.075 |
| CNN + ECA, mean pool | 0.435, 0.472, 0.475 | 0.461 ± 0.022 |
| CNN + ECA, attention pool | 0.413, 0.385, 0.430 | 0.409 ± 0.022 |
| CNN + CBAM, mean pool | 0.439, 0.509, 0.448 | 0.465 ± 0.038 |
| CNN + CBAM, attention pool | 0.508, 0.529, 0.453 | 0.497 ± 0.039 |
| small ResNet + ECA, mean pool | 0.443, 0.366, 0.437 | 0.416 ± 0.043 |
| small ResNet + ECA, attention pool | 0.444, 0.426, 0.414 | 0.428 ± 0.015 |
| small ResNet + CBAM, mean pool | 0.386, 0.464, 0.484 | 0.445 ± 0.052 |
| small ResNet + CBAM, attention pool | 0.466, 0.472, 0.499 | 0.479 ± 0.017 |

Median best epoch: CNN, mean pool 2, CNN, attention pool 2, CNN, mean ∥ max pool 2, small ResNet, mean pool 2, small ResNet, attention pool 2, small ResNet, mean ∥ max pool 2, small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 1, ResNet-18, mean pool 3, ResNet-18, attention pool 4, CNN + ECA, mean pool 1, CNN + ECA, attention pool 1, CNN + CBAM, mean pool 7, CNN + CBAM, attention pool 1, small ResNet + ECA, mean pool 2, small ResNet + ECA, attention pool 2, small ResNet + CBAM, mean pool 1, small ResNet + CBAM, attention pool 2.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.004 [-0.065, +0.073] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.099 [+0.051, +0.156] |
| CNN + ECA, mean pool | CNN, mean pool | +0.042 [-0.001, +0.096] |
| CNN + ECA, attention pool | CNN, attention pool | -0.015 [-0.075, +0.046] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.030 [-0.060, +0.124] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.140 [+0.083, +0.196] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.096 [+0.032, +0.170] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.004 [-0.067, +0.063] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.090 [+0.042, +0.120] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.088 [+0.028, +0.148] |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| CNN, mean pool | 0.482 [0.431, 0.531] | -0.069 [-0.121, -0.019] | 0.478 | 0.580 | 0.450 | 0.525 | 0.500 |
| CNN, attention pool | 0.454 [0.408, 0.495] | -0.097 [-0.151, -0.043] | 0.436 | 0.512 | 0.469 | 0.508 | 0.500 |
| CNN, mean ∥ max pool | 0.450 [0.398, 0.505] | -0.102 [-0.162, -0.045] | 0.465 | 0.512 | 0.425 | 0.437 | 0.556 |
| small ResNet, mean pool | 0.482 [0.435, 0.528] | -0.069 [-0.123, -0.017] | 0.535 | 0.564 | 0.481 | 0.474 | 0.514 |
| small ResNet, attention pool | 0.483 [0.437, 0.528] | -0.069 [-0.121, -0.016] | 0.522 | 0.552 | 0.523 | 0.456 | 0.485 |
| small ResNet, mean ∥ max pool | 0.458 [0.420, 0.496] | -0.093 [-0.156, -0.033] | 0.577 | 0.528 | 0.500 | 0.437 | 0.500 |
| small ResNet (no max pool), mean pool | 0.479 [0.428, 0.528] | -0.073 [-0.126, -0.023] | 0.535 | 0.517 | 0.462 | 0.491 | 0.514 |
| small ResNet (no max pool), attention pool | 0.498 [0.446, 0.545] | -0.053 [-0.104, -0.001] | 0.518 | 0.540 | 0.535 | 0.508 | 0.500 |
| ResNet-18, mean pool | 0.509 [0.451, 0.564] | -0.042 [-0.120, +0.029] | 0.539 | 0.500 | 0.572 | 0.497 | 0.586 |
| ResNet-18, attention pool | 0.511 [0.474, 0.549] | -0.040 [-0.103, +0.027] | 0.493 | 0.500 | 0.561 | 0.552 | 0.586 |
| CNN + ECA, mean pool | 0.465 [0.420, 0.508] | -0.087 [-0.142, -0.033] | 0.535 | 0.509 | 0.450 | 0.474 | 0.500 |
| CNN + ECA, attention pool | 0.490 [0.438, 0.540] | -0.062 [-0.115, -0.011] | 0.493 | 0.533 | 0.511 | 0.508 | 0.514 |
| CNN + CBAM, mean pool | 0.521 [0.462, 0.581] | -0.031 [-0.083, +0.018] | 0.518 | 0.514 | 0.577 | 0.553 | 0.498 |
| CNN + CBAM, attention pool | 0.468 [0.415, 0.521] | -0.084 [-0.136, -0.033] | 0.535 | 0.545 | 0.481 | 0.443 | 0.498 |
| small ResNet + ECA, mean pool | 0.461 [0.420, 0.497] | -0.091 [-0.146, -0.035] | 0.498 | 0.524 | 0.469 | 0.508 | 0.500 |
| small ResNet + ECA, attention pool | 0.458 [0.414, 0.500] | -0.094 [-0.147, -0.040] | 0.509 | 0.521 | 0.469 | 0.456 | 0.500 |
| small ResNet + CBAM, mean pool | 0.465 [0.416, 0.510] | -0.087 [-0.139, -0.031] | 0.504 | 0.490 | 0.469 | 0.508 | 0.500 |
| small ResNet + CBAM, attention pool | 0.464 [0.418, 0.510] | -0.088 [-0.149, -0.027] | 0.478 | 0.524 | 0.481 | 0.502 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.443, 0.516, 0.483 | 0.480 ± 0.037 |
| CNN, attention pool | 0.439, 0.474, 0.466 | 0.459 ± 0.018 |
| CNN, mean ∥ max pool | 0.517, 0.465, 0.469 | 0.484 ± 0.029 |
| small ResNet, mean pool | 0.475, 0.500, 0.447 | 0.474 ± 0.027 |
| small ResNet, attention pool | 0.503, 0.499, 0.451 | 0.484 ± 0.029 |
| small ResNet, mean ∥ max pool | 0.472, 0.467, 0.444 | 0.461 ± 0.015 |
| small ResNet (no max pool), mean pool | 0.487, 0.511, 0.466 | 0.488 ± 0.022 |
| small ResNet (no max pool), attention pool | 0.477, 0.495, 0.523 | 0.498 ± 0.023 |
| ResNet-18, mean pool | 0.475, 0.472, 0.469 | 0.472 ± 0.003 |
| ResNet-18, attention pool | 0.480, 0.504, 0.434 | 0.473 ± 0.036 |
| CNN + ECA, mean pool | 0.490, 0.493, 0.468 | 0.484 ± 0.014 |
| CNN + ECA, attention pool | 0.474, 0.494, 0.526 | 0.498 ± 0.026 |
| CNN + CBAM, mean pool | 0.527, 0.517, 0.516 | 0.520 ± 0.006 |
| CNN + CBAM, attention pool | 0.479, 0.497, 0.458 | 0.478 ± 0.019 |
| small ResNet + ECA, mean pool | 0.500, 0.453, 0.479 | 0.477 ± 0.024 |
| small ResNet + ECA, attention pool | 0.469, 0.470, 0.452 | 0.464 ± 0.010 |
| small ResNet + CBAM, mean pool | 0.477, 0.474, 0.451 | 0.467 ± 0.015 |
| small ResNet + CBAM, attention pool | 0.503, 0.481, 0.477 | 0.487 ± 0.014 |

Median best epoch: CNN, mean pool 3, CNN, attention pool 2, CNN, mean ∥ max pool 5, small ResNet, mean pool 4, small ResNet, attention pool 5, small ResNet, mean ∥ max pool 5, small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 5, ResNet-18, mean pool 11, ResNet-18, attention pool 11, CNN + ECA, mean pool 13, CNN + ECA, attention pool 4, CNN + CBAM, mean pool 6, CNN + CBAM, attention pool 6, small ResNet + ECA, mean pool 2, small ResNet + ECA, attention pool 2, small ResNet + CBAM, mean pool 4, small ResNet + CBAM, attention pool 3.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.033 [-0.083, +0.018] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.024 [-0.069, +0.020] |
| CNN + ECA, mean pool | CNN, mean pool | -0.017 [-0.045, +0.012] |
| CNN + ECA, attention pool | CNN, attention pool | +0.035 [-0.003, +0.076] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.038 [-0.004, +0.083] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.014 [-0.029, +0.057] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | -0.022 [-0.053, +0.008] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.025 [-0.059, +0.009] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | -0.017 [-0.049, +0.013] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.019 [-0.064, +0.025] |
