# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| CNN + CBAM, mean pool | 0.575 [0.541, 0.608] | +0.051 [-0.020, +0.119] | 0.554 | 0.513 | 0.500 | 0.438 | 0.500 |
| CNN + CBAM, attention pool | 0.538 [0.501, 0.578] | +0.013 [-0.062, +0.085] | 0.500 | 0.498 | 0.446 | 0.500 | 0.330 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.573, 0.605, 0.505 | 0.561 ± 0.051 |
| CNN + CBAM, attention pool | 0.560, 0.585, 0.546 | 0.564 ± 0.020 |

Median best epoch: CNN + CBAM, mean pool 7, CNN + CBAM, attention pool 14.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| CNN + CBAM, mean pool | 0.486 [0.449, 0.521] | -0.020 [-0.086, +0.048] | 0.648 | 0.490 | 0.444 | 0.500 | 0.417 |
| CNN + CBAM, attention pool | 0.474 [0.427, 0.522] | -0.031 [-0.102, +0.042] | 0.394 | 0.505 | 0.437 | 0.560 | 0.497 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.504, 0.478, 0.498 | 0.493 ± 0.014 |
| CNN + CBAM, attention pool | 0.522, 0.527, 0.453 | 0.501 ± 0.042 |

Median best epoch: CNN + CBAM, mean pool 6, CNN + CBAM, attention pool 9.
