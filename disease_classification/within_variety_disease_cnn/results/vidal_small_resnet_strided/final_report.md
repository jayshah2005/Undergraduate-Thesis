# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| small ResNet (no max pool), mean pool | 0.448 [0.390, 0.501] | -0.109 [-0.203, -0.025] | 0.611 | 0.409 | 0.670 | 0.500 | 0.510 |
| small ResNet (no max pool), attention pool | 0.488 [0.437, 0.536] | -0.068 [-0.135, -0.001] | 0.558 | 0.458 | 0.681 | 0.500 | 0.562 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.463, 0.445, 0.389 | 0.432 ± 0.038 |
| small ResNet (no max pool), attention pool | 0.458, 0.474, 0.472 | 0.468 ± 0.009 |

Median best epoch: small ResNet (no max pool), mean pool 13, small ResNet (no max pool), attention pool 4.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| small ResNet (no max pool), mean pool | 0.509 [0.453, 0.562] | -0.038 [-0.109, +0.035] | 0.586 | 0.620 | 0.505 | 0.467 | 0.498 |
| small ResNet (no max pool), attention pool | 0.519 [0.467, 0.570] | -0.028 [-0.098, +0.044] | 0.671 | 0.604 | 0.469 | 0.467 | 0.541 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.514, 0.528, 0.522 | 0.521 ± 0.007 |
| small ResNet (no max pool), attention pool | 0.539, 0.504, 0.465 | 0.503 ± 0.037 |

Median best epoch: small ResNet (no max pool), mean pool 4, small ResNet (no max pool), attention pool 6.
