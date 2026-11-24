# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN + ECA, mean pool | 0.573 [0.545, 0.601] | +0.037 [-0.029, +0.100] | 0.463 | 0.500 | 0.501 | 0.500 | 0.612 |
| CNN + ECA, attention pool | 0.580 [0.548, 0.608] | +0.043 [-0.024, +0.108] | 0.481 | 0.500 | 0.486 | 0.500 | 0.613 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.567, 0.557, 0.522 | 0.549 ± 0.024 |
| CNN + ECA, attention pool | 0.587, 0.574, 0.534 | 0.565 ± 0.027 |

Median best epoch: CNN + ECA, mean pool 8, CNN + ECA, attention pool 7.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN + ECA, mean pool | 0.536 [0.491, 0.580] | -0.069 [-0.128, -0.009] | 0.505 | 0.550 | 0.709 | 0.510 | 0.500 |
| CNN + ECA, attention pool | 0.578 [0.523, 0.632] | -0.027 [-0.088, +0.030] | 0.518 | 0.517 | 0.731 | 0.537 | 0.613 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.519, 0.483, 0.540 | 0.514 ± 0.028 |
| CNN + ECA, attention pool | 0.539, 0.529, 0.598 | 0.555 ± 0.037 |

Median best epoch: CNN + ECA, mean pool 8, CNN + ECA, attention pool 9.
