# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet + CBAM, mean pool | 0.610 [0.591, 0.629] | +0.039 [+0.004, +0.072] | 0.585 | 0.689 | 0.516 | 0.555 | 0.672 |
| small ResNet + CBAM, attention pool | 0.583 [0.565, 0.601] | +0.012 [-0.023, +0.046] | 0.533 | 0.680 | 0.509 | 0.453 | 0.706 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet + CBAM, mean pool | 0.527 | 0.488 | 0.500 |
| small ResNet + CBAM, attention pool | 0.501 | 0.431 | 0.510 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.595, 0.597, 0.554 | 0.582 ± 0.024 |
| small ResNet + CBAM, attention pool | 0.565, 0.577, 0.561 | 0.567 ± 0.008 |

Median best epoch: small ResNet + CBAM, mean pool 14, small ResNet + CBAM, attention pool 15.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| small ResNet + CBAM, mean pool | 0.686 [0.666, 0.704] | +0.039 [+0.013, +0.065] | 0.678 | 0.691 | 0.705 | 0.679 | 0.691 |
| small ResNet + CBAM, attention pool | 0.678 [0.660, 0.695] | +0.031 [+0.006, +0.057] | 0.613 | 0.674 | 0.737 | 0.673 | 0.672 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet + CBAM, mean pool | 0.482 | 0.572 | 0.531 | 0.539 |
| small ResNet + CBAM, attention pool | 0.514 | 0.533 | 0.546 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.674, 0.670, 0.666 | 0.670 ± 0.004 |
| small ResNet + CBAM, attention pool | 0.679, 0.682, 0.665 | 0.675 ± 0.009 |

Median best epoch: small ResNet + CBAM, mean pool 13, small ResNet + CBAM, attention pool 8.
