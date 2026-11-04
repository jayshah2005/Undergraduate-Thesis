# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.481 [0.448, 0.512] | — | 0.507 | 0.446 | 0.526 | 0.485 | 0.481 |
| CNN, mean pool | 0.447 [0.419, 0.477] | -0.033 [-0.070, +0.004] | 0.575 | 0.521 | 0.403 | 0.450 | 0.457 |
| CNN, attention pool | 0.494 [0.468, 0.520] | +0.013 [-0.025, +0.051] | 0.573 | 0.506 | 0.456 | 0.463 | 0.496 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.457, 0.491, 0.445 | 0.464 ± 0.024 |
| CNN, attention pool | 0.526, 0.464, 0.492 | 0.494 ± 0.031 |

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN, mean pool | 0.370 [0.344, 0.398] | -0.049 [-0.097, +0.002] | 0.500 | 0.519 | 0.304 | 0.400 | 0.517 |
| CNN, attention pool | 0.440 [0.407, 0.472] | +0.022 [-0.028, +0.074] | 0.500 | 0.500 | 0.357 | 0.425 | 0.543 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.405, 0.434, 0.365 | 0.401 ± 0.035 |
| CNN, attention pool | 0.509, 0.369, 0.440 | 0.439 ± 0.070 |

Median best epoch: CNN, mean pool 3, CNN, attention pool 2.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN, mean pool | 0.525 [0.475, 0.576] | -0.018 [-0.077, +0.037] | 0.649 | 0.524 | 0.502 | 0.500 | 0.397 |
| CNN, attention pool | 0.548 [0.500, 0.595] | +0.005 [-0.060, +0.071] | 0.646 | 0.513 | 0.554 | 0.500 | 0.449 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.508, 0.548, 0.524 | 0.527 ± 0.020 |
| CNN, attention pool | 0.543, 0.559, 0.544 | 0.549 ± 0.009 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5.
