# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| CNN + ECA, mean pool | 0.603 [0.579, 0.625] | +0.031 [-0.005, +0.066] | 0.637 | 0.672 | 0.540 | 0.503 | 0.628 |
| CNN + ECA, attention pool | 0.590 [0.568, 0.611] | +0.018 [-0.018, +0.054] | 0.581 | 0.687 | 0.500 | 0.494 | 0.628 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| CNN + ECA, mean pool | 0.480 | 0.477 | 0.492 |
| CNN + ECA, attention pool | 0.478 | 0.503 | 0.488 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.592, 0.553, 0.585 | 0.577 ± 0.021 |
| CNN + ECA, attention pool | 0.573, 0.582, 0.556 | 0.570 ± 0.013 |

Median best epoch: CNN + ECA, mean pool 17, CNN + ECA, attention pool 23.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| CNN + ECA, mean pool | 0.686 [0.669, 0.702] | +0.039 [+0.014, +0.064] | 0.629 | 0.675 | 0.768 | 0.667 | 0.690 |
| CNN + ECA, attention pool | 0.681 [0.663, 0.698] | +0.034 [+0.009, +0.058] | 0.665 | 0.693 | 0.719 | 0.667 | 0.692 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| CNN + ECA, mean pool | 0.489 | 0.545 | 0.569 | 0.519 |
| CNN + ECA, attention pool | 0.483 | 0.586 | 0.539 | 0.498 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.685, 0.643, 0.659 | 0.662 ± 0.021 |
| CNN + ECA, attention pool | 0.658, 0.664, 0.650 | 0.657 ± 0.007 |

Median best epoch: CNN + ECA, mean pool 10, CNN + ECA, attention pool 10.
