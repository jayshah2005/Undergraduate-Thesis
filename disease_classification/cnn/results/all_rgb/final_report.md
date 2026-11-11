# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| CNN, mean pool | 0.565 [0.535, 0.593] | +0.021 [-0.004, +0.045] | 0.557 | 0.627 | 0.546 | 0.546 | 0.578 |
| CNN, attention pool | 0.580 [0.552, 0.607] | +0.036 [+0.010, +0.064] | 0.583 | 0.627 | 0.568 | 0.559 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| CNN, mean pool | 0.462 | 0.436 | 0.493 |
| CNN, attention pool | 0.540 | 0.423 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.578, 0.548, 0.556 | 0.561 ± 0.015 |
| CNN, attention pool | 0.569, 0.572, 0.589 | 0.576 ± 0.011 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 2.

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | — | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| CNN, mean pool | 0.650 [0.631, 0.668] | +0.009 [-0.012, +0.030] | 0.636 | 0.635 | 0.651 | 0.679 | 0.657 |
| CNN, attention pool | 0.667 [0.647, 0.686] | +0.026 [+0.003, +0.048] | 0.659 | 0.647 | 0.666 | 0.688 | 0.674 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.458 | 0.499 | 0.567 | 0.542 |
| CNN, mean pool | 0.497 | 0.489 | 0.517 | 0.471 |
| CNN, attention pool | 0.503 | 0.502 | 0.534 | 0.503 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.677, 0.655, 0.656 | 0.663 ± 0.012 |
| CNN, attention pool | 0.658, 0.672, 0.668 | 0.666 ± 0.007 |

Median best epoch: CNN, mean pool 15, CNN, attention pool 11.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | — | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| CNN, mean pool | 0.614 [0.592, 0.635] | +0.030 [+0.001, +0.056] | 0.527 | 0.567 | 0.606 | 0.645 | 0.651 |
| CNN, attention pool | 0.597 [0.575, 0.619] | +0.013 [-0.016, +0.041] | 0.491 | 0.543 | 0.606 | 0.627 | 0.618 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.449 | 0.495 | 0.542 | 0.541 |
| CNN, mean pool | 0.487 | 0.563 | 0.500 | 0.536 |
| CNN, attention pool | 0.521 | 0.538 | 0.500 | 0.493 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.616, 0.604, 0.613 | 0.611 ± 0.006 |
| CNN, attention pool | 0.599, 0.602, 0.599 | 0.600 ± 0.002 |

Median best epoch: CNN, mean pool 21, CNN, attention pool 11.
