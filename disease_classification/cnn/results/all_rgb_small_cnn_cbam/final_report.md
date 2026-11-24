# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| CNN + CBAM, mean pool | 0.590 [0.561, 0.620] | +0.048 [+0.017, +0.080] | 0.608 | 0.636 | 0.597 | 0.571 | 0.529 |
| CNN + CBAM, attention pool | 0.602 [0.575, 0.630] | +0.059 [+0.032, +0.088] | 0.594 | 0.621 | 0.632 | 0.556 | 0.569 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| CNN + CBAM, mean pool | 0.534 | 0.446 | 0.494 |
| CNN + CBAM, attention pool | 0.505 | 0.504 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.569, 0.588, 0.609 | 0.588 ± 0.020 |
| CNN + CBAM, attention pool | 0.599, 0.603, 0.585 | 0.596 ± 0.010 |

Median best epoch: CNN + CBAM, mean pool 8, CNN + CBAM, attention pool 7.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| CNN + CBAM, mean pool | 0.678 [0.658, 0.698] | +0.034 [+0.012, +0.056] | 0.656 | 0.667 | 0.718 | 0.683 | 0.678 |
| CNN + CBAM, attention pool | 0.672 [0.651, 0.693] | +0.028 [+0.004, +0.051] | 0.657 | 0.676 | 0.677 | 0.677 | 0.681 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| CNN + CBAM, mean pool | 0.501 | 0.529 | 0.537 | 0.522 |
| CNN + CBAM, attention pool | 0.477 | 0.524 | 0.533 | 0.534 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.685, 0.665, 0.655 | 0.668 ± 0.015 |
| CNN + CBAM, attention pool | 0.681, 0.653, 0.662 | 0.665 ± 0.014 |

Median best epoch: CNN + CBAM, mean pool 15, CNN + CBAM, attention pool 17.
