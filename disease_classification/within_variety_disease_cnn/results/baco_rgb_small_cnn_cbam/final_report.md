# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN + CBAM, mean pool | 0.511 [0.466, 0.558] | +0.093 [+0.032, +0.156] | 0.537 | 0.509 | 0.382 | 0.450 | 0.556 |
| CNN + CBAM, attention pool | 0.557 [0.519, 0.594] | +0.139 [+0.075, +0.205] | 0.500 | 0.486 | 0.486 | 0.475 | 0.476 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.421, 0.546, 0.585 | 0.517 ± 0.085 |
| CNN + CBAM, attention pool | 0.419, 0.598, 0.574 | 0.530 ± 0.097 |

Median best epoch: CNN + CBAM, mean pool 5, CNN + CBAM, attention pool 2.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN + CBAM, mean pool | 0.566 [0.517, 0.615] | +0.023 [-0.029, +0.072] | 0.521 | 0.542 | 0.521 | 0.580 | 0.524 |
| CNN + CBAM, attention pool | 0.539 [0.490, 0.589] | -0.005 [-0.053, +0.048] | 0.450 | 0.560 | 0.531 | 0.620 | 0.412 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.513, 0.531, 0.544 | 0.529 ± 0.016 |
| CNN + CBAM, attention pool | 0.539, 0.542, 0.546 | 0.543 ± 0.004 |

Median best epoch: CNN + CBAM, mean pool 11, CNN + CBAM, attention pool 9.
