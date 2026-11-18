# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| small ResNet (no max pool), mean pool | 0.555 [0.525, 0.585] | +0.011 [-0.014, +0.035] | 0.569 | 0.603 | 0.524 | 0.551 | 0.542 |
| small ResNet (no max pool), attention pool | 0.574 [0.546, 0.601] | +0.030 [+0.003, +0.055] | 0.586 | 0.613 | 0.522 | 0.551 | 0.584 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| small ResNet (no max pool), mean pool | 0.468 | 0.427 | 0.487 |
| small ResNet (no max pool), attention pool | 0.499 | 0.423 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.570, 0.508, 0.571 | 0.549 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.569, 0.558, 0.567 | 0.565 ± 0.006 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 7.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | — | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| small ResNet (no max pool), mean pool | 0.605 [0.579, 0.630] | +0.001 [-0.018, +0.021] | 0.609 | 0.544 | 0.675 | 0.583 | 0.650 |
| small ResNet (no max pool), attention pool | 0.608 [0.582, 0.633] | +0.004 [-0.015, +0.023] | 0.597 | 0.547 | 0.686 | 0.583 | 0.649 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.434 | 0.610 | 0.551 |
| small ResNet (no max pool), mean pool | 0.477 | 0.576 | 0.549 |
| small ResNet (no max pool), attention pool | 0.467 | 0.594 | 0.546 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.578, 0.603, 0.601 | 0.594 ± 0.014 |
| small ResNet (no max pool), attention pool | 0.595, 0.608, 0.605 | 0.603 ± 0.007 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 5.
