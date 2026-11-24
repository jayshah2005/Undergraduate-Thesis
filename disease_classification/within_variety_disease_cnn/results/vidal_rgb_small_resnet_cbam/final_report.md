# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| small ResNet + CBAM, mean pool | 0.457 [0.406, 0.500] | +0.018 [-0.056, +0.094] | 0.450 | 0.432 | 0.511 | 0.488 | 0.500 |
| small ResNet + CBAM, attention pool | 0.480 [0.412, 0.547] | +0.041 [-0.031, +0.112] | 0.450 | 0.409 | 0.500 | 0.565 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.386, 0.464, 0.484 | 0.445 ± 0.052 |
| small ResNet + CBAM, attention pool | 0.466, 0.472, 0.499 | 0.479 ± 0.017 |

Median best epoch: small ResNet + CBAM, mean pool 1, small ResNet + CBAM, attention pool 2.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| small ResNet + CBAM, mean pool | 0.465 [0.416, 0.510] | -0.087 [-0.139, -0.031] | 0.504 | 0.490 | 0.469 | 0.508 | 0.500 |
| small ResNet + CBAM, attention pool | 0.464 [0.418, 0.510] | -0.088 [-0.149, -0.027] | 0.478 | 0.524 | 0.481 | 0.502 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.477, 0.474, 0.451 | 0.467 ± 0.015 |
| small ResNet + CBAM, attention pool | 0.503, 0.481, 0.477 | 0.487 ± 0.014 |

Median best epoch: small ResNet + CBAM, mean pool 4, small ResNet + CBAM, attention pool 3.
