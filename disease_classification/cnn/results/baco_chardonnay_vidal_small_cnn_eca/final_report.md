# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| CNN + ECA, mean pool | 0.580 [0.556, 0.603] | -0.020 [-0.052, +0.012] | 0.530 | 0.574 | 0.647 | 0.529 | 0.638 |
| CNN + ECA, attention pool | 0.609 [0.589, 0.627] | +0.009 [-0.020, +0.037] | 0.597 | 0.574 | 0.613 | 0.577 | 0.635 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| CNN + ECA, mean pool | 0.441 | 0.548 | 0.530 |
| CNN + ECA, attention pool | 0.503 | 0.553 | 0.544 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.580, 0.596, 0.598 | 0.591 ± 0.010 |
| CNN + ECA, attention pool | 0.607, 0.596, 0.613 | 0.605 ± 0.008 |

Median best epoch: CNN + ECA, mean pool 9, CNN + ECA, attention pool 12.
