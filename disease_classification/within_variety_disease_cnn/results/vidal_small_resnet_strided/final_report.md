# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | — | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |
| small ResNet (no max pool), mean pool | 0.433 [0.384, 0.471] | -0.111 [-0.182, -0.047] | 0.568 | 0.386 | 0.667 | 0.500 | 0.510 |
| small ResNet (no max pool), attention pool | 0.493 [0.420, 0.567] | -0.051 [-0.118, +0.018] | 0.542 | 0.473 | 0.677 | 0.500 | 0.750 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.500, 0.420, 0.427 | 0.449 ± 0.044 |
| small ResNet (no max pool), attention pool | 0.480, 0.393, 0.456 | 0.443 ± 0.045 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 6.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | — | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |
| small ResNet (no max pool), mean pool | 0.512 [0.467, 0.558] | -0.040 [-0.113, +0.032] | 0.595 | 0.567 | 0.537 | 0.439 | 0.587 |
| small ResNet (no max pool), attention pool | 0.506 [0.451, 0.558] | -0.046 [-0.117, +0.025] | 0.554 | 0.633 | 0.475 | 0.527 | 0.557 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.499, 0.483, 0.535 | 0.506 ± 0.026 |
| small ResNet (no max pool), attention pool | 0.497, 0.490, 0.535 | 0.508 ± 0.024 |

Median best epoch: small ResNet (no max pool), mean pool 7, small ResNet (no max pool), attention pool 4.
