# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.561, 0.645] | — | 0.493 | 0.581 | 0.723 | 0.586 | 0.588 |
| CNN, mean pool | 0.558 [0.523, 0.592] | -0.046 [-0.077, -0.016] | 0.574 | 0.517 | 0.599 | 0.524 | 0.464 |
| CNN, attention pool | 0.576 [0.539, 0.611] | -0.028 [-0.061, +0.005] | 0.504 | 0.536 | 0.633 | 0.546 | 0.541 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.563, 0.540, 0.556 | 0.553 ± 0.012 |
| CNN, attention pool | 0.584, 0.572, 0.549 | 0.568 ± 0.018 |

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN, mean pool | 0.531 [0.487, 0.575] | -0.057 [-0.105, -0.009] | 0.567 | 0.500 | 0.486 | 0.475 | 0.465 |
| CNN, attention pool | 0.569 [0.525, 0.613] | -0.019 [-0.068, +0.034] | 0.470 | 0.598 | 0.500 | 0.525 | 0.607 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.529, 0.521, 0.522 | 0.524 ± 0.004 |
| CNN, attention pool | 0.562, 0.572, 0.498 | 0.544 ± 0.040 |

Median best epoch: CNN, mean pool 11, CNN, attention pool 6.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN, mean pool | 0.584 [0.529, 0.642] | -0.035 [-0.075, +0.002] | 0.582 | 0.533 | 0.713 | 0.573 | 0.463 |
| CNN, attention pool | 0.583 [0.525, 0.641] | -0.037 [-0.078, +0.005] | 0.537 | 0.475 | 0.765 | 0.567 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.597, 0.559, 0.590 | 0.582 ± 0.020 |
| CNN, attention pool | 0.606, 0.572, 0.601 | 0.593 ± 0.018 |

Median best epoch: CNN, mean pool 7, CNN, attention pool 9.
