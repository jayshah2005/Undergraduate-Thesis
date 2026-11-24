# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| small ResNet + CBAM, mean pool | 0.607 [0.585, 0.631] | +0.008 [-0.022, +0.038] | 0.557 | 0.592 | 0.651 | 0.584 | 0.627 |
| small ResNet + CBAM, attention pool | 0.623 [0.597, 0.648] | +0.023 [-0.010, +0.057] | 0.576 | 0.635 | 0.679 | 0.584 | 0.645 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| small ResNet + CBAM, mean pool | 0.481 | 0.564 | 0.562 |
| small ResNet + CBAM, attention pool | 0.529 | 0.579 | 0.554 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + CBAM, mean pool | 0.595, 0.606, 0.631 | 0.611 ± 0.018 |
| small ResNet + CBAM, attention pool | 0.592, 0.614, 0.593 | 0.600 ± 0.012 |

Median best epoch: small ResNet + CBAM, mean pool 15, small ResNet + CBAM, attention pool 9.
