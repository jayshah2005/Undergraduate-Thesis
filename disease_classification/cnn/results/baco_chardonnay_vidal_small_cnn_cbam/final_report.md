# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| CNN + CBAM, mean pool | 0.623 [0.602, 0.644] | +0.023 [-0.007, +0.053] | 0.590 | 0.633 | 0.643 | 0.571 | 0.632 |
| CNN + CBAM, attention pool | 0.589 [0.565, 0.612] | -0.011 [-0.043, +0.020] | 0.538 | 0.580 | 0.626 | 0.597 | 0.630 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| CNN + CBAM, mean pool | 0.527 | 0.560 | 0.548 |
| CNN + CBAM, attention pool | 0.463 | 0.548 | 0.527 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.588, 0.611, 0.614 | 0.604 ± 0.014 |
| CNN + CBAM, attention pool | 0.580, 0.559, 0.622 | 0.587 ± 0.032 |

Median best epoch: CNN + CBAM, mean pool 6, CNN + CBAM, attention pool 12.
