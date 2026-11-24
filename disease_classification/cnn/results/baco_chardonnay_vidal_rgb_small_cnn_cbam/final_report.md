# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| CNN + CBAM, mean pool | 0.599 [0.573, 0.625] | -0.011 [-0.034, +0.012] | 0.576 | 0.604 | 0.649 | 0.514 | 0.662 |
| CNN + CBAM, attention pool | 0.608 [0.582, 0.634] | -0.002 [-0.022, +0.018] | 0.607 | 0.588 | 0.665 | 0.549 | 0.662 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| CNN + CBAM, mean pool | 0.482 | 0.571 | 0.526 |
| CNN + CBAM, attention pool | 0.468 | 0.594 | 0.535 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.587, 0.598, 0.616 | 0.600 ± 0.014 |
| CNN + CBAM, attention pool | 0.597, 0.591, 0.613 | 0.600 ± 0.011 |

Median best epoch: CNN + CBAM, mean pool 14, CNN + CBAM, attention pool 12.
