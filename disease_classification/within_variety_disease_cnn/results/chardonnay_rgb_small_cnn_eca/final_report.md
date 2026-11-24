# Disease: whole-leaf CNN on Chardonnay, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | — | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN + ECA, mean pool | 0.546 [0.498, 0.598] | -0.041 [-0.097, +0.015] | 0.554 | 0.561 | 0.531 | 0.450 | 0.510 |
| CNN + ECA, attention pool | 0.563 [0.515, 0.611] | -0.024 [-0.078, +0.025] | 0.566 | 0.494 | 0.488 | 0.475 | 0.548 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.514, 0.581, 0.487 | 0.527 ± 0.049 |
| CNN + ECA, attention pool | 0.532, 0.551, 0.605 | 0.563 ± 0.038 |

Median best epoch: CNN + ECA, mean pool 20, CNN + ECA, attention pool 17.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | — | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN + ECA, mean pool | 0.600 [0.552, 0.649] | -0.020 [-0.071, +0.029] | 0.500 | 0.508 | 0.686 | 0.583 | 0.475 |
| CNN + ECA, attention pool | 0.600 [0.552, 0.649] | -0.020 [-0.070, +0.029] | 0.500 | 0.508 | 0.700 | 0.567 | 0.475 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.632, 0.586, 0.626 | 0.615 ± 0.025 |
| CNN + ECA, attention pool | 0.602, 0.576, 0.543 | 0.574 ± 0.029 |

Median best epoch: CNN + ECA, mean pool 10, CNN + ECA, attention pool 10.
