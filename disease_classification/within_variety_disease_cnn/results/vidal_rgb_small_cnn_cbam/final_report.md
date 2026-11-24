# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| CNN + CBAM, mean pool | 0.445 [0.398, 0.480] | +0.006 [-0.097, +0.108] | 0.487 | 0.477 | 0.564 | 0.405 | 0.365 |
| CNN + CBAM, attention pool | 0.550 [0.507, 0.604] | +0.111 [+0.029, +0.192] | 0.500 | 0.455 | 0.500 | 0.530 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.439, 0.509, 0.448 | 0.465 ± 0.038 |
| CNN + CBAM, attention pool | 0.508, 0.529, 0.453 | 0.497 ± 0.039 |

Median best epoch: CNN + CBAM, mean pool 7, CNN + CBAM, attention pool 1.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| CNN + CBAM, mean pool | 0.521 [0.462, 0.581] | -0.031 [-0.083, +0.018] | 0.518 | 0.514 | 0.577 | 0.553 | 0.498 |
| CNN + CBAM, attention pool | 0.468 [0.415, 0.521] | -0.084 [-0.136, -0.033] | 0.535 | 0.545 | 0.481 | 0.443 | 0.498 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.527, 0.517, 0.516 | 0.520 ± 0.006 |
| CNN + CBAM, attention pool | 0.479, 0.497, 0.458 | 0.478 ± 0.019 |

Median best epoch: CNN + CBAM, mean pool 6, CNN + CBAM, attention pool 6.
