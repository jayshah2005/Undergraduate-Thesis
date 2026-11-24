# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| CNN + ECA, mean pool | 0.453 [0.388, 0.518] | -0.103 [-0.196, -0.014] | 0.611 | 0.504 | 0.606 | 0.500 | 0.500 |
| CNN + ECA, attention pool | 0.517 [0.455, 0.579] | -0.039 [-0.125, +0.038] | 0.718 | 0.527 | 0.500 | 0.500 | 0.760 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.438, 0.463, 0.443 | 0.448 ± 0.014 |
| CNN + ECA, attention pool | 0.546, 0.425, 0.424 | 0.465 ± 0.070 |

Median best epoch: CNN + ECA, mean pool 11, CNN + ECA, attention pool 7.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| CNN + ECA, mean pool | 0.502 [0.456, 0.547] | -0.046 [-0.118, +0.027] | 0.632 | 0.528 | 0.547 | 0.439 | 0.498 |
| CNN + ECA, attention pool | 0.510 [0.469, 0.552] | -0.037 [-0.104, +0.032] | 0.684 | 0.500 | 0.469 | 0.498 | 0.571 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.474, 0.517, 0.608 | 0.533 ± 0.068 |
| CNN + ECA, attention pool | 0.517, 0.492, 0.534 | 0.514 ± 0.021 |

Median best epoch: CNN + ECA, mean pool 8, CNN + ECA, attention pool 5.
