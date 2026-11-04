# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.477 [0.434, 0.520] | — | 0.531 | 0.473 | 0.369 | 0.472 | 0.591 |
| CNN, mean pool | 0.447 [0.408, 0.483] | -0.029 [-0.080, +0.017] | 0.496 | 0.449 | 0.525 | 0.403 | 0.451 |
| CNN, attention pool | 0.442 [0.412, 0.474] | -0.035 [-0.074, +0.002] | 0.523 | 0.493 | 0.488 | 0.454 | 0.520 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.466, 0.505, 0.460 | 0.477 ± 0.024 |
| CNN, attention pool | 0.445, 0.436, 0.442 | 0.441 ± 0.005 |

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| CNN, mean pool | 0.396 [0.327, 0.460] | -0.020 [-0.104, +0.069] | 0.475 | 0.398 | 0.500 | 0.315 | 0.344 |
| CNN, attention pool | 0.407 [0.362, 0.464] | -0.009 [-0.078, +0.060] | 0.500 | 0.477 | 0.500 | 0.399 | 0.510 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.420, 0.504, 0.431 | 0.452 ± 0.046 |
| CNN, attention pool | 0.422, 0.387, 0.440 | 0.416 ± 0.027 |

Median best epoch: CNN, mean pool 2, CNN, attention pool 2.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| CNN, mean pool | 0.499 [0.445, 0.550] | -0.039 [-0.091, +0.013] | 0.518 | 0.500 | 0.550 | 0.491 | 0.558 |
| CNN, attention pool | 0.477 [0.435, 0.520] | -0.061 [-0.113, -0.006] | 0.547 | 0.508 | 0.475 | 0.508 | 0.529 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.513, 0.506, 0.489 | 0.503 ± 0.012 |
| CNN, attention pool | 0.468, 0.485, 0.443 | 0.465 ± 0.021 |

Median best epoch: CNN, mean pool 8, CNN, attention pool 2.
