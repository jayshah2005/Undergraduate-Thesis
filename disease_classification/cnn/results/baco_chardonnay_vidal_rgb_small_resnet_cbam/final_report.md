# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| small ResNet + CBAM, mean pool | 0.618 [0.594, 0.642] | +0.009 [-0.013, +0.031] | 0.604 | 0.582 | 0.717 | 0.583 | 0.644 |
| small ResNet + CBAM, attention pool | 0.589 [0.562, 0.617] | -0.020 [-0.042, +0.001] | 0.551 | 0.579 | 0.661 | 0.550 | 0.646 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| small ResNet + CBAM, mean pool | 0.495 | 0.583 | 0.544 |
| small ResNet + CBAM, attention pool | 0.454 | 0.565 | 0.528 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.610, 0.637, 0.610 | 0.619 ± 0.016 |
| small ResNet + CBAM, attention pool | 0.592, 0.605, 0.583 | 0.593 ± 0.011 |

Median best epoch: small ResNet + CBAM, mean pool 20, small ResNet + CBAM, attention pool 9.
