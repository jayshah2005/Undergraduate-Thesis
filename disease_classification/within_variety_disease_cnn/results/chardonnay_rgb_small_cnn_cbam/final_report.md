# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN + CBAM, mean pool | 0.528 [0.478, 0.580] | -0.060 [-0.132, +0.009] | 0.554 | 0.438 | 0.445 | 0.450 | 0.500 |
| CNN + CBAM, attention pool | 0.510 [0.467, 0.554] | -0.077 [-0.140, -0.020] | 0.581 | 0.524 | 0.487 | 0.400 | 0.500 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.557, 0.505, 0.501 | 0.521 ± 0.031 |
| CNN + CBAM, attention pool | 0.489, 0.541, 0.451 | 0.494 ± 0.045 |

Median best epoch: CNN + CBAM, mean pool 11, CNN + CBAM, attention pool 8.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN + CBAM, mean pool | 0.588 [0.534, 0.644] | -0.031 [-0.080, +0.015] | 0.553 | 0.508 | 0.725 | 0.557 | 0.475 |
| CNN + CBAM, attention pool | 0.563 [0.504, 0.620] | -0.057 [-0.103, -0.010] | 0.514 | 0.533 | 0.584 | 0.557 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.571, 0.600, 0.559 | 0.577 ± 0.021 |
| CNN + CBAM, attention pool | 0.589, 0.573, 0.567 | 0.576 ± 0.011 |

Median best epoch: CNN + CBAM, mean pool 8, CNN + CBAM, attention pool 12.
