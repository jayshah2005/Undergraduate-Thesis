# Disease: whole-leaf CNN on Baco, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN + ECA, mean pool | 0.486 [0.459, 0.515] | +0.068 [+0.015, +0.120] | 0.500 | 0.519 | 0.532 | 0.450 | 0.500 |
| CNN + ECA, attention pool | 0.489 [0.458, 0.519] | +0.070 [+0.025, +0.115] | 0.500 | 0.515 | 0.464 | 0.400 | 0.500 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.392, 0.486, 0.499 | 0.459 ± 0.058 |
| CNN + ECA, attention pool | 0.510, 0.443, 0.403 | 0.452 ± 0.054 |

Median best epoch: CNN + ECA, mean pool 3, CNN + ECA, attention pool 1.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN + ECA, mean pool | 0.530 [0.478, 0.580] | -0.013 [-0.060, +0.033] | 0.569 | 0.524 | 0.509 | 0.580 | 0.396 |
| CNN + ECA, attention pool | 0.528 [0.479, 0.578] | -0.015 [-0.068, +0.037] | 0.573 | 0.534 | 0.460 | 0.600 | 0.401 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.516, 0.572, 0.535 | 0.541 ± 0.028 |
| CNN + ECA, attention pool | 0.537, 0.533, 0.555 | 0.541 ± 0.012 |

Median best epoch: CNN + ECA, mean pool 10, CNN + ECA, attention pool 8.
