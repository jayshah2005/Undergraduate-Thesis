# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| small ResNet + CBAM, mean pool | 0.580 [0.550, 0.610] | +0.037 [+0.010, +0.066] | 0.609 | 0.615 | 0.594 | 0.544 | 0.554 |
| small ResNet + CBAM, attention pool | 0.579 [0.551, 0.606] | +0.037 [+0.007, +0.068] | 0.613 | 0.617 | 0.540 | 0.580 | 0.549 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| small ResNet + CBAM, mean pool | 0.498 | 0.439 | 0.508 |
| small ResNet + CBAM, attention pool | 0.503 | 0.462 | 0.491 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.591, 0.561, 0.569 | 0.574 ± 0.016 |
| small ResNet + CBAM, attention pool | 0.582, 0.579, 0.591 | 0.584 ± 0.006 |

Median best epoch: small ResNet + CBAM, mean pool 3, small ResNet + CBAM, attention pool 10.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| small ResNet + CBAM, mean pool | 0.676 [0.654, 0.696] | +0.031 [+0.008, +0.053] | 0.672 | 0.657 | 0.694 | 0.696 | 0.653 |
| small ResNet + CBAM, attention pool | 0.673 [0.653, 0.693] | +0.029 [+0.006, +0.052] | 0.641 | 0.662 | 0.696 | 0.677 | 0.699 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| small ResNet + CBAM, mean pool | 0.495 | 0.508 | 0.524 | 0.552 |
| small ResNet + CBAM, attention pool | 0.486 | 0.504 | 0.531 | 0.532 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.676, 0.674, 0.677 | 0.676 ± 0.001 |
| small ResNet + CBAM, attention pool | 0.675, 0.672, 0.670 | 0.672 ± 0.002 |

Median best epoch: small ResNet + CBAM, mean pool 16, small ResNet + CBAM, attention pool 14.
