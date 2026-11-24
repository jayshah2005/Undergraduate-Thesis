# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| small ResNet + CBAM, mean pool | 0.451 [0.387, 0.507] | -0.105 [-0.190, -0.022] | 0.508 | 0.564 | 0.543 | 0.500 | 0.510 |
| small ResNet + CBAM, attention pool | 0.453 [0.406, 0.507] | -0.103 [-0.185, -0.029] | 0.500 | 0.473 | 0.511 | 0.562 | 0.500 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.463, 0.483, 0.518 | 0.488 ± 0.028 |
| small ResNet + CBAM, attention pool | 0.428, 0.458, 0.434 | 0.440 ± 0.016 |

Median best epoch: small ResNet + CBAM, mean pool 5, small ResNet + CBAM, attention pool 2.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| small ResNet + CBAM, mean pool | 0.521 [0.480, 0.562] | -0.026 [-0.090, +0.035] | 0.711 | 0.556 | 0.500 | 0.543 | 0.513 |
| small ResNet + CBAM, attention pool | 0.509 [0.469, 0.547] | -0.038 [-0.103, +0.025] | 0.618 | 0.500 | 0.512 | 0.567 | 0.555 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.492, 0.524, 0.498 | 0.505 ± 0.017 |
| small ResNet + CBAM, attention pool | 0.502, 0.503, 0.510 | 0.505 ± 0.004 |

Median best epoch: small ResNet + CBAM, mean pool 9, small ResNet + CBAM, attention pool 7.
