# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.515 [0.473, 0.558] | — | 0.508 | 0.556 | 0.486 | 0.561 | 0.475 |
| CNN, mean pool | 0.525 [0.500, 0.552] | +0.010 [-0.039, +0.058] | 0.521 | 0.500 | 0.488 | 0.550 | 0.486 |
| CNN, attention pool | 0.517 [0.491, 0.542] | +0.002 [-0.044, +0.049] | 0.526 | 0.451 | 0.500 | 0.481 | 0.470 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.513, 0.536, 0.505 | 0.518 ± 0.016 |
| CNN, attention pool | 0.508, 0.545, 0.538 | 0.531 ± 0.020 |

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| CNN, mean pool | 0.546 [0.522, 0.571] | +0.021 [-0.043, +0.084] | 0.524 | 0.500 | 0.525 | 0.500 | 0.500 |
| CNN, attention pool | 0.529 [0.502, 0.556] | +0.005 [-0.060, +0.068] | 0.500 | 0.494 | 0.500 | 0.463 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.555, 0.576, 0.518 | 0.550 ± 0.029 |
| CNN, attention pool | 0.505, 0.577, 0.554 | 0.546 ± 0.037 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 10.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| CNN, mean pool | 0.504 [0.456, 0.550] | -0.001 [-0.074, +0.074] | 0.519 | 0.500 | 0.451 | 0.600 | 0.471 |
| CNN, attention pool | 0.505 [0.465, 0.547] | -0.001 [-0.068, +0.069] | 0.552 | 0.409 | 0.500 | 0.500 | 0.441 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.471, 0.496, 0.491 | 0.486 ± 0.013 |
| CNN, attention pool | 0.511, 0.514, 0.523 | 0.516 ± 0.006 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5.
