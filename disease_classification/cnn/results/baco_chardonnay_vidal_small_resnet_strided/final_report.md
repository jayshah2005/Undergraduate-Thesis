# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | +0.021 [-0.016, +0.055] | 0.572 | 0.662 | 0.495 | 0.513 | 0.688 |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | +0.030 [-0.009, +0.065] | 0.504 | 0.684 | 0.571 | 0.578 | 0.625 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| small ResNet (no max pool), mean pool | 0.528 | 0.458 | 0.498 |
| small ResNet (no max pool), attention pool | 0.538 | 0.467 | 0.544 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.597, 0.556, 0.589 | 0.581 ± 0.021 |
| small ResNet (no max pool), attention pool | 0.577, 0.573, 0.594 | 0.582 ± 0.011 |

Median best epoch: small ResNet (no max pool), mean pool 12, small ResNet (no max pool), attention pool 15.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | — | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| small ResNet (no max pool), mean pool | 0.616 [0.592, 0.639] | +0.019 [-0.011, +0.050] | 0.575 | 0.613 | 0.617 | 0.594 | 0.657 |
| small ResNet (no max pool), attention pool | 0.610 [0.589, 0.630] | +0.013 [-0.017, +0.041] | 0.572 | 0.579 | 0.681 | 0.576 | 0.639 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.460 | 0.548 | 0.546 |
| small ResNet (no max pool), mean pool | 0.538 | 0.551 | 0.525 |
| small ResNet (no max pool), attention pool | 0.474 | 0.538 | 0.571 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.618, 0.584, 0.596 | 0.599 ± 0.017 |
| small ResNet (no max pool), attention pool | 0.581, 0.595, 0.596 | 0.591 ± 0.008 |

Median best epoch: small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 7.
