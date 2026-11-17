# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| CNN, mean pool | 0.396 [0.327, 0.460] | -0.020 [-0.104, +0.069] | 0.475 | 0.398 | 0.500 | 0.315 | 0.344 |
| CNN, attention pool | 0.407 [0.362, 0.464] | -0.009 [-0.078, +0.060] | 0.500 | 0.477 | 0.500 | 0.399 | 0.510 |
| small ResNet, mean pool | 0.376 [0.320, 0.422] | -0.040 [-0.109, +0.031] | 0.450 | 0.420 | 0.500 | 0.440 | 0.562 |
| small ResNet, attention pool | 0.340 [0.291, 0.376] | -0.076 [-0.153, +0.002] | 0.475 | 0.386 | 0.500 | 0.429 | 0.073 |
| small ResNet (no max pool), mean pool | 0.353 [0.284, 0.433] | -0.062 [-0.173, +0.060] | 0.468 | 0.424 | 0.500 | 0.396 | 0.510 |
| small ResNet (no max pool), attention pool | 0.458 [0.442, 0.473] | +0.042 [-0.036, +0.122] | 0.512 | 0.500 | 0.500 | 0.500 | 0.708 |
| ResNet-18, mean pool | 0.360 [0.313, 0.413] | -0.056 [-0.160, +0.056] | 0.271 | 0.477 | 0.562 | 0.500 | 0.573 |
| ResNet-18, attention pool | 0.384 [0.367, 0.402] | -0.031 [-0.113, +0.051] | 0.523 | 0.489 | 0.521 | 0.417 | 0.104 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.420, 0.504, 0.431 | 0.452 ± 0.046 |
| CNN, attention pool | 0.422, 0.387, 0.440 | 0.416 ± 0.027 |
| small ResNet, mean pool | 0.391, 0.391, 0.367 | 0.383 ± 0.014 |
| small ResNet, attention pool | 0.438, 0.402, 0.378 | 0.406 ± 0.030 |
| small ResNet (no max pool), mean pool | 0.424, 0.342, 0.447 | 0.404 ± 0.055 |
| small ResNet (no max pool), attention pool | 0.398, 0.476, 0.398 | 0.424 ± 0.045 |
| ResNet-18, mean pool | 0.478, 0.347, 0.396 | 0.407 ± 0.066 |
| ResNet-18, attention pool | 0.431, 0.396, 0.462 | 0.430 ± 0.033 |

Median best epoch: CNN, mean pool 2, CNN, attention pool 2, small ResNet, mean pool 2, small ResNet, attention pool 3, small ResNet (no max pool), mean pool 1, small ResNet (no max pool), attention pool 1, ResNet-18, mean pool 2, ResNet-18, attention pool 4.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| CNN, mean pool | 0.499 [0.445, 0.550] | -0.039 [-0.091, +0.013] | 0.518 | 0.500 | 0.550 | 0.491 | 0.558 |
| CNN, attention pool | 0.477 [0.435, 0.520] | -0.061 [-0.113, -0.006] | 0.547 | 0.508 | 0.475 | 0.508 | 0.529 |
| small ResNet, mean pool | 0.484 [0.441, 0.526] | -0.054 [-0.109, +0.000] | 0.558 | 0.558 | 0.475 | 0.508 | 0.500 |
| small ResNet, attention pool | 0.510 [0.459, 0.564] | -0.027 [-0.076, +0.024] | 0.545 | 0.500 | 0.625 | 0.467 | 0.529 |
| small ResNet (no max pool), mean pool | 0.503 [0.449, 0.557] | -0.035 [-0.089, +0.020] | 0.502 | 0.508 | 0.600 | 0.502 | 0.525 |
| small ResNet (no max pool), attention pool | 0.480 [0.438, 0.523] | -0.057 [-0.109, -0.003] | 0.556 | 0.517 | 0.513 | 0.491 | 0.485 |
| ResNet-18, mean pool | 0.436 [0.386, 0.485] | -0.102 [-0.166, -0.042] | 0.452 | 0.500 | 0.475 | 0.411 | 0.583 |
| ResNet-18, attention pool | 0.471 [0.419, 0.525] | -0.067 [-0.132, -0.003] | 0.445 | 0.500 | 0.500 | 0.568 | 0.538 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.513, 0.506, 0.489 | 0.503 ± 0.012 |
| CNN, attention pool | 0.468, 0.485, 0.443 | 0.465 ± 0.021 |
| small ResNet, mean pool | 0.462, 0.501, 0.451 | 0.472 ± 0.026 |
| small ResNet, attention pool | 0.515, 0.510, 0.462 | 0.496 ± 0.029 |
| small ResNet (no max pool), mean pool | 0.518, 0.502, 0.453 | 0.491 ± 0.034 |
| small ResNet (no max pool), attention pool | 0.481, 0.521, 0.512 | 0.505 ± 0.021 |
| ResNet-18, mean pool | 0.454, 0.481, 0.438 | 0.457 ± 0.022 |
| ResNet-18, attention pool | 0.429, 0.459, 0.479 | 0.456 ± 0.025 |

Median best epoch: CNN, mean pool 8, CNN, attention pool 2, small ResNet, mean pool 3, small ResNet, attention pool 10, small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 2, ResNet-18, mean pool 8, ResNet-18, attention pool 11.
