# Disease: whole-leaf CNN on Chardonnay, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| small ResNet (no max pool), mean pool | 0.572 [0.537, 0.608] | +0.036 [-0.032, +0.104] | 0.525 | 0.500 | 0.486 | 0.487 | 0.620 |
| small ResNet (no max pool), attention pool | 0.531 [0.507, 0.555] | -0.006 [-0.069, +0.055] | 0.503 | 0.500 | 0.442 | 0.463 | 0.500 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.564, 0.547, 0.564 | 0.558 ± 0.010 |
| small ResNet (no max pool), attention pool | 0.531, 0.556, 0.465 | 0.518 ± 0.047 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 9.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| small ResNet (no max pool), mean pool | 0.568 [0.515, 0.622] | -0.037 [-0.102, +0.028] | 0.559 | 0.600 | 0.715 | 0.510 | 0.525 |
| small ResNet (no max pool), attention pool | 0.570 [0.517, 0.623] | -0.035 [-0.101, +0.030] | 0.508 | 0.508 | 0.734 | 0.510 | 0.575 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.553, 0.530, 0.536 | 0.540 ± 0.012 |
| small ResNet (no max pool), attention pool | 0.539, 0.530, 0.577 | 0.549 ± 0.025 |

Median best epoch: small ResNet (no max pool), mean pool 14, small ResNet (no max pool), attention pool 12.
