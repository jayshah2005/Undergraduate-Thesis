# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| CNN, mean pool | 0.415 [0.337, 0.494] | -0.024 [-0.125, +0.087] | 0.475 | 0.436 | 0.500 | 0.351 | 0.458 |
| CNN, attention pool | 0.410 [0.367, 0.461] | -0.029 [-0.129, +0.074] | 0.500 | 0.477 | 0.500 | 0.363 | 0.510 |
| small ResNet, mean pool | 0.368 [0.323, 0.400] | -0.071 [-0.149, +0.009] | 0.487 | 0.375 | 0.500 | 0.381 | 0.500 |
| small ResNet, attention pool | 0.392 [0.347, 0.421] | -0.047 [-0.127, +0.037] | 0.475 | 0.386 | 0.500 | 0.500 | 0.510 |
| small ResNet (no max pool), mean pool | 0.431 [0.359, 0.507] | -0.008 [-0.112, +0.098] | 0.487 | 0.508 | 0.500 | 0.458 | 0.510 |
| small ResNet (no max pool), attention pool | 0.411 [0.400, 0.424] | -0.028 [-0.109, +0.060] | 0.500 | 0.500 | 0.500 | 0.500 | 0.125 |
| ResNet-18, mean pool | 0.395 [0.339, 0.447] | -0.044 [-0.127, +0.033] | 0.458 | 0.398 | 0.564 | 0.500 | 0.542 |
| ResNet-18, attention pool | 0.378 [0.331, 0.430] | -0.061 [-0.163, +0.041] | 0.399 | 0.477 | 0.521 | 0.452 | 0.552 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.442, 0.488, 0.423 | 0.451 ± 0.033 |
| CNN, attention pool | 0.415, 0.365, 0.417 | 0.399 ± 0.029 |
| small ResNet, mean pool | 0.416, 0.422, 0.404 | 0.414 ± 0.009 |
| small ResNet, attention pool | 0.383, 0.399, 0.376 | 0.386 ± 0.012 |
| small ResNet (no max pool), mean pool | 0.433, 0.392, 0.457 | 0.427 ± 0.033 |
| small ResNet (no max pool), attention pool | 0.409, 0.469, 0.432 | 0.437 ± 0.030 |
| ResNet-18, mean pool | 0.405, 0.334, 0.462 | 0.400 ± 0.064 |
| ResNet-18, attention pool | 0.418, 0.302, 0.442 | 0.387 ± 0.075 |

Median best epoch: CNN, mean pool 2, CNN, attention pool 2, small ResNet, mean pool 2, small ResNet, attention pool 2, small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 1, ResNet-18, mean pool 3, ResNet-18, attention pool 4.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| CNN, mean pool | 0.482 [0.431, 0.531] | -0.069 [-0.121, -0.019] | 0.478 | 0.580 | 0.450 | 0.525 | 0.500 |
| CNN, attention pool | 0.454 [0.408, 0.495] | -0.097 [-0.151, -0.043] | 0.436 | 0.512 | 0.469 | 0.508 | 0.500 |
| small ResNet, mean pool | 0.482 [0.435, 0.528] | -0.069 [-0.123, -0.017] | 0.535 | 0.564 | 0.481 | 0.474 | 0.514 |
| small ResNet, attention pool | 0.483 [0.437, 0.528] | -0.069 [-0.121, -0.016] | 0.522 | 0.552 | 0.523 | 0.456 | 0.485 |
| small ResNet (no max pool), mean pool | 0.479 [0.428, 0.528] | -0.073 [-0.126, -0.023] | 0.535 | 0.517 | 0.462 | 0.491 | 0.514 |
| small ResNet (no max pool), attention pool | 0.498 [0.446, 0.545] | -0.053 [-0.104, -0.001] | 0.518 | 0.540 | 0.535 | 0.508 | 0.500 |
| ResNet-18, mean pool | 0.509 [0.451, 0.564] | -0.042 [-0.120, +0.029] | 0.539 | 0.500 | 0.572 | 0.497 | 0.586 |
| ResNet-18, attention pool | 0.511 [0.474, 0.549] | -0.040 [-0.103, +0.027] | 0.493 | 0.500 | 0.561 | 0.552 | 0.586 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.443, 0.516, 0.483 | 0.480 ± 0.037 |
| CNN, attention pool | 0.439, 0.474, 0.466 | 0.459 ± 0.018 |
| small ResNet, mean pool | 0.475, 0.500, 0.447 | 0.474 ± 0.027 |
| small ResNet, attention pool | 0.503, 0.499, 0.451 | 0.484 ± 0.029 |
| small ResNet (no max pool), mean pool | 0.487, 0.511, 0.466 | 0.488 ± 0.022 |
| small ResNet (no max pool), attention pool | 0.477, 0.495, 0.523 | 0.498 ± 0.023 |
| ResNet-18, mean pool | 0.475, 0.472, 0.469 | 0.472 ± 0.003 |
| ResNet-18, attention pool | 0.480, 0.504, 0.434 | 0.473 ± 0.036 |

Median best epoch: CNN, mean pool 3, CNN, attention pool 2, small ResNet, mean pool 4, small ResNet, attention pool 5, small ResNet (no max pool), mean pool 3, small ResNet (no max pool), attention pool 5, ResNet-18, mean pool 11, ResNet-18, attention pool 11.
