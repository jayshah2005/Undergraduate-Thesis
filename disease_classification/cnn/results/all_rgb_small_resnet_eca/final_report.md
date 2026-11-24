# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| small ResNet + ECA, mean pool | 0.551 [0.522, 0.578] | +0.008 [-0.019, +0.034] | 0.573 | 0.609 | 0.575 | 0.506 | 0.547 |
| small ResNet + ECA, attention pool | 0.612 [0.584, 0.637] | +0.069 [+0.040, +0.098] | 0.594 | 0.646 | 0.559 | 0.627 | 0.550 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| small ResNet + ECA, mean pool | 0.468 | 0.420 | 0.481 |
| small ResNet + ECA, attention pool | 0.538 | 0.499 | 0.492 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.583, 0.536, 0.549 | 0.556 ± 0.024 |
| small ResNet + ECA, attention pool | 0.585, 0.562, 0.594 | 0.580 ± 0.016 |

Median best epoch: small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 4.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| small ResNet + ECA, mean pool | 0.679 [0.658, 0.699] | +0.035 [+0.014, +0.056] | 0.663 | 0.670 | 0.695 | 0.690 | 0.689 |
| small ResNet + ECA, attention pool | 0.676 [0.655, 0.697] | +0.031 [+0.010, +0.052] | 0.658 | 0.633 | 0.718 | 0.692 | 0.695 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| small ResNet + ECA, mean pool | 0.510 | 0.497 | 0.534 | 0.549 |
| small ResNet + ECA, attention pool | 0.507 | 0.483 | 0.534 | 0.559 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.667, 0.664, 0.665 | 0.665 ± 0.001 |
| small ResNet + ECA, attention pool | 0.676, 0.654, 0.670 | 0.666 ± 0.011 |

Median best epoch: small ResNet + ECA, mean pool 17, small ResNet + ECA, attention pool 12.
