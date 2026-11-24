# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| small ResNet + ECA, mean pool | 0.554 [0.522, 0.587] | +0.018 [-0.050, +0.089] | 0.473 | 0.484 | 0.500 | 0.500 | 0.518 |
| small ResNet + ECA, attention pool | 0.533 [0.498, 0.566] | -0.003 [-0.070, +0.061] | 0.463 | 0.494 | 0.486 | 0.463 | 0.525 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.553, 0.561, 0.559 | 0.558 ± 0.004 |
| small ResNet + ECA, attention pool | 0.481, 0.591, 0.526 | 0.533 ± 0.055 |

Median best epoch: small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 5.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| small ResNet + ECA, mean pool | 0.577 [0.526, 0.627] | -0.028 [-0.093, +0.036] | 0.505 | 0.542 | 0.627 | 0.510 | 0.587 |
| small ResNet + ECA, attention pool | 0.608 [0.556, 0.659] | +0.003 [-0.058, +0.060] | 0.603 | 0.550 | 0.731 | 0.526 | 0.625 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.536, 0.565, 0.597 | 0.566 ± 0.030 |
| small ResNet + ECA, attention pool | 0.545, 0.543, 0.541 | 0.543 ± 0.002 |

Median best epoch: small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 4.
