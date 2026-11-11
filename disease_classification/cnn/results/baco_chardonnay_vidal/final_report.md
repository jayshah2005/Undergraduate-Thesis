# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.611, 0.630, 0.591 | 0.611 ± 0.019 |
| CNN, attention pool | 0.612, 0.607, 0.611 | 0.610 ± 0.002 |

Median best epoch: CNN, mean pool 20, CNN, attention pool 16.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | — | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| CNN, mean pool | 0.608 [0.585, 0.631] | +0.012 [-0.019, +0.043] | 0.598 | 0.589 | 0.609 | 0.569 | 0.653 |
| CNN, attention pool | 0.605 [0.583, 0.626] | +0.008 [-0.022, +0.037] | 0.559 | 0.579 | 0.681 | 0.571 | 0.632 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.460 | 0.548 | 0.546 |
| CNN, mean pool | 0.483 | 0.565 | 0.559 |
| CNN, attention pool | 0.452 | 0.577 | 0.562 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.593, 0.612, 0.596 | 0.600 ± 0.010 |
| CNN, attention pool | 0.586, 0.571, 0.615 | 0.590 ± 0.023 |

Median best epoch: CNN, mean pool 12, CNN, attention pool 6.
