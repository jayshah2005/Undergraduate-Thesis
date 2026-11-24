# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| CNN + CBAM, mean pool | 0.610 [0.591, 0.628] | +0.038 [+0.000, +0.073] | 0.625 | 0.691 | 0.551 | 0.533 | 0.683 |
| CNN + CBAM, attention pool | 0.624 [0.604, 0.643] | +0.052 [+0.016, +0.088] | 0.583 | 0.656 | 0.663 | 0.455 | 0.665 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| CNN + CBAM, mean pool | 0.497 | 0.456 | 0.496 |
| CNN + CBAM, attention pool | 0.497 | 0.508 | 0.525 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.594, 0.567, 0.606 | 0.589 ± 0.020 |
| CNN + CBAM, attention pool | 0.609, 0.570, 0.608 | 0.596 ± 0.022 |

Median best epoch: CNN + CBAM, mean pool 14, CNN + CBAM, attention pool 14.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| CNN + CBAM, mean pool | 0.702 [0.683, 0.720] | +0.055 [+0.028, +0.081] | 0.668 | 0.687 | 0.740 | 0.679 | 0.736 |
| CNN + CBAM, attention pool | 0.701 [0.680, 0.721] | +0.054 [+0.028, +0.082] | 0.672 | 0.729 | 0.715 | 0.683 | 0.680 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| CNN + CBAM, mean pool | 0.489 | 0.522 | 0.558 | 0.581 |
| CNN + CBAM, attention pool | 0.503 | 0.537 | 0.584 | 0.542 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + CBAM, mean pool | 0.670, 0.682, 0.687 | 0.680 ± 0.009 |
| CNN + CBAM, attention pool | 0.669, 0.687, 0.678 | 0.678 ± 0.009 |

Median best epoch: CNN + CBAM, mean pool 19, CNN + CBAM, attention pool 15.
