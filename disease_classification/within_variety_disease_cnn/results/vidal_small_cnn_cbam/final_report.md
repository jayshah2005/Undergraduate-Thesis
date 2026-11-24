# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| CNN + CBAM, mean pool | 0.425 [0.365, 0.490] | -0.131 [-0.240, -0.023] | 0.550 | 0.508 | 0.553 | 0.488 | 0.521 |
| CNN + CBAM, attention pool | 0.468 [0.412, 0.532] | -0.089 [-0.189, +0.012] | 0.457 | 0.538 | 0.521 | 0.488 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.472, 0.419, 0.428 | 0.440 ± 0.028 |
| CNN + CBAM, attention pool | 0.510, 0.519, 0.496 | 0.508 ± 0.011 |

Median best epoch: CNN + CBAM, mean pool 14, CNN + CBAM, attention pool 6.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| CNN + CBAM, mean pool | 0.562 [0.514, 0.608] | +0.014 [-0.058, +0.084] | 0.695 | 0.500 | 0.621 | 0.511 | 0.538 |
| CNN + CBAM, attention pool | 0.476 [0.433, 0.517] | -0.071 [-0.139, -0.002] | 0.526 | 0.500 | 0.505 | 0.439 | 0.585 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.519, 0.537, 0.565 | 0.540 ± 0.024 |
| CNN + CBAM, attention pool | 0.486, 0.488, 0.511 | 0.495 ± 0.014 |

Median best epoch: CNN + CBAM, mean pool 10, CNN + CBAM, attention pool 5.
