# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| small ResNet + CBAM, mean pool | 0.567 [0.539, 0.595] | +0.030 [-0.040, +0.094] | 0.522 | 0.500 | 0.500 | 0.463 | 0.539 |
| small ResNet + CBAM, attention pool | 0.517 [0.482, 0.553] | -0.019 [-0.091, +0.048] | 0.522 | 0.500 | 0.427 | 0.400 | 0.469 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.563, 0.548, 0.594 | 0.568 ± 0.023 |
| small ResNet + CBAM, attention pool | 0.539, 0.547, 0.540 | 0.542 ± 0.004 |

Median best epoch: small ResNet + CBAM, mean pool 5, small ResNet + CBAM, attention pool 8.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| small ResNet + CBAM, mean pool | 0.603 [0.551, 0.653] | -0.002 [-0.058, +0.053] | 0.550 | 0.575 | 0.765 | 0.510 | 0.562 |
| small ResNet + CBAM, attention pool | 0.558 [0.507, 0.608] | -0.047 [-0.107, +0.010] | 0.494 | 0.533 | 0.538 | 0.526 | 0.575 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.586, 0.525, 0.614 | 0.575 ± 0.045 |
| small ResNet + CBAM, attention pool | 0.506, 0.572, 0.561 | 0.546 ± 0.035 |

Median best epoch: small ResNet + CBAM, mean pool 6, small ResNet + CBAM, attention pool 4.
