# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| CNN + ECA, mean pool | 0.609 [0.583, 0.634] | -0.001 [-0.019, +0.018] | 0.603 | 0.587 | 0.662 | 0.571 | 0.655 |
| CNN + ECA, attention pool | 0.610 [0.586, 0.635] | +0.001 [-0.021, +0.022] | 0.598 | 0.583 | 0.663 | 0.576 | 0.662 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| CNN + ECA, mean pool | 0.463 | 0.591 | 0.544 |
| CNN + ECA, attention pool | 0.478 | 0.588 | 0.535 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.604, 0.604, 0.607 | 0.605 ± 0.002 |
| CNN + ECA, attention pool | 0.596, 0.597, 0.610 | 0.601 ± 0.008 |

Median best epoch: CNN + ECA, mean pool 6, CNN + ECA, attention pool 4.
