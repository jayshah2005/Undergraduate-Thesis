# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| small ResNet (no max pool), mean pool | 0.353 [0.284, 0.433] | -0.062 [-0.173, +0.060] | 0.468 | 0.424 | 0.500 | 0.396 | 0.510 |
| small ResNet (no max pool), attention pool | 0.458 [0.442, 0.473] | +0.042 [-0.036, +0.122] | 0.512 | 0.500 | 0.500 | 0.500 | 0.708 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.424, 0.342, 0.447 | 0.404 ± 0.055 |
| small ResNet (no max pool), attention pool | 0.398, 0.476, 0.398 | 0.424 ± 0.045 |

Median best epoch: small ResNet (no max pool), mean pool 1, small ResNet (no max pool), attention pool 1.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| small ResNet (no max pool), mean pool | 0.503 [0.449, 0.557] | -0.035 [-0.089, +0.020] | 0.502 | 0.508 | 0.600 | 0.502 | 0.525 |
| small ResNet (no max pool), attention pool | 0.480 [0.438, 0.523] | -0.057 [-0.109, -0.003] | 0.556 | 0.517 | 0.513 | 0.491 | 0.485 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.518, 0.502, 0.453 | 0.491 ± 0.034 |
| small ResNet (no max pool), attention pool | 0.481, 0.521, 0.512 | 0.505 ± 0.021 |

Median best epoch: small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 2.
