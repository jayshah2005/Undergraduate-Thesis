# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| small ResNet + ECA, mean pool | 0.609 [0.586, 0.632] | +0.009 [-0.021, +0.040] | 0.576 | 0.593 | 0.654 | 0.568 | 0.633 |
| small ResNet + ECA, attention pool | 0.626 [0.605, 0.646] | +0.027 [-0.003, +0.054] | 0.590 | 0.603 | 0.670 | 0.598 | 0.626 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| small ResNet + ECA, mean pool | 0.492 | 0.563 | 0.543 |
| small ResNet + ECA, attention pool | 0.521 | 0.543 | 0.576 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet + ECA, mean pool | 0.604, 0.609, 0.594 | 0.602 ± 0.008 |
| small ResNet + ECA, attention pool | 0.632, 0.613, 0.619 | 0.622 ± 0.010 |

Median best epoch: small ResNet + ECA, mean pool 13, small ResNet + ECA, attention pool 9.
