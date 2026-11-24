# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| CNN + ECA, mean pool | 0.543 [0.518, 0.571] | +0.018 [-0.052, +0.088] | 0.500 | 0.500 | 0.525 | 0.512 | 0.500 |
| CNN + ECA, attention pool | 0.588 [0.569, 0.607] | +0.064 [-0.001, +0.124] | 0.513 | 0.500 | 0.500 | 0.463 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.529, 0.500, 0.539 | 0.523 ± 0.020 |
| CNN + ECA, attention pool | 0.526, 0.473, 0.558 | 0.519 ± 0.043 |

Median best epoch: CNN + ECA, mean pool 16, CNN + ECA, attention pool 14.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| CNN + ECA, mean pool | 0.472 [0.424, 0.521] | -0.034 [-0.106, +0.039] | 0.477 | 0.451 | 0.411 | 0.480 | 0.420 |
| CNN + ECA, attention pool | 0.517 [0.480, 0.554] | +0.012 [-0.055, +0.078] | 0.484 | 0.539 | 0.483 | 0.500 | 0.494 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.446, 0.477, 0.536 | 0.486 ± 0.046 |
| CNN + ECA, attention pool | 0.538, 0.562, 0.521 | 0.540 ± 0.021 |

Median best epoch: CNN + ECA, mean pool 11, CNN + ECA, attention pool 6.
