# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet + ECA, mean pool | 0.607 [0.589, 0.625] | +0.035 [+0.001, +0.068] | 0.574 | 0.699 | 0.491 | 0.523 | 0.657 |
| small ResNet + ECA, attention pool | 0.580 [0.560, 0.599] | +0.008 [-0.028, +0.042] | 0.586 | 0.691 | 0.518 | 0.503 | 0.603 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet + ECA, mean pool | 0.527 | 0.485 | 0.483 |
| small ResNet + ECA, attention pool | 0.489 | 0.450 | 0.526 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.626, 0.638, 0.555 | 0.607 ± 0.045 |
| small ResNet + ECA, attention pool | 0.562, 0.585, 0.566 | 0.571 ± 0.012 |

Median best epoch: small ResNet + ECA, mean pool 22, small ResNet + ECA, attention pool 10.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| small ResNet + ECA, mean pool | 0.683 [0.666, 0.701] | +0.037 [+0.011, +0.062] | 0.648 | 0.673 | 0.730 | 0.683 | 0.685 |
| small ResNet + ECA, attention pool | 0.680 [0.662, 0.699] | +0.033 [+0.007, +0.060] | 0.587 | 0.701 | 0.725 | 0.681 | 0.682 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet + ECA, mean pool | 0.463 | 0.533 | 0.547 | 0.529 |
| small ResNet + ECA, attention pool | 0.482 | 0.545 | 0.565 | 0.512 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.698, 0.678, 0.680 | 0.685 ± 0.011 |
| small ResNet + ECA, attention pool | 0.681, 0.668, 0.670 | 0.673 ± 0.007 |

Median best epoch: small ResNet + ECA, mean pool 17, small ResNet + ECA, attention pool 14.
