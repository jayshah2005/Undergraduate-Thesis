# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN, mean pool | 0.591 [0.555, 0.628] | +0.055 [-0.014, +0.119] | 0.615 | 0.500 | 0.501 | 0.475 | 0.572 |
| CNN, attention pool | 0.559 [0.531, 0.587] | +0.023 [-0.038, +0.079] | 0.500 | 0.500 | 0.471 | 0.475 | 0.522 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.576, 0.546, 0.554 | 0.558 ± 0.016 |
| CNN, attention pool | 0.558, 0.531, 0.572 | 0.554 ± 0.021 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 7.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN, mean pool | 0.523 [0.475, 0.570] | -0.082 [-0.143, -0.020] | 0.514 | 0.575 | 0.627 | 0.494 | 0.500 |
| CNN, attention pool | 0.519 [0.472, 0.566] | -0.086 [-0.152, -0.023] | 0.537 | 0.600 | 0.668 | 0.500 | 0.500 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.518, 0.563, 0.543 | 0.541 ± 0.022 |
| CNN, attention pool | 0.498, 0.504, 0.541 | 0.514 ± 0.024 |

Median best epoch: CNN, mean pool 5, CNN, attention pool 6.
