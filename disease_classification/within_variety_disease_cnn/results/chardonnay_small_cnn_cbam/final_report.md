# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN + CBAM, mean pool | 0.529 [0.493, 0.562] | -0.007 [-0.075, +0.057] | 0.466 | 0.397 | 0.471 | 0.487 | 0.520 |
| CNN + CBAM, attention pool | 0.565 [0.533, 0.601] | +0.028 [-0.036, +0.092] | 0.500 | 0.494 | 0.502 | 0.500 | 0.523 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.581, 0.512, 0.573 | 0.555 ± 0.038 |
| CNN + CBAM, attention pool | 0.555, 0.542, 0.562 | 0.553 ± 0.010 |

Median best epoch: CNN + CBAM, mean pool 8, CNN + CBAM, attention pool 8.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN + CBAM, mean pool | 0.594 [0.541, 0.646] | -0.011 [-0.068, +0.043] | 0.510 | 0.525 | 0.806 | 0.526 | 0.537 |
| CNN + CBAM, attention pool | 0.566 [0.514, 0.619] | -0.039 [-0.101, +0.021] | 0.474 | 0.500 | 0.725 | 0.494 | 0.562 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.534, 0.543, 0.595 | 0.558 ± 0.033 |
| CNN + CBAM, attention pool | 0.549, 0.533, 0.565 | 0.549 ± 0.016 |

Median best epoch: CNN + CBAM, mean pool 7, CNN + CBAM, attention pool 7.
