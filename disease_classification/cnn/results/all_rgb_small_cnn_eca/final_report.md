# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| CNN + ECA, mean pool | 0.566 [0.537, 0.594] | +0.024 [-0.001, +0.047] | 0.567 | 0.627 | 0.560 | 0.543 | 0.541 |
| CNN + ECA, attention pool | 0.587 [0.558, 0.614] | +0.045 [+0.018, +0.071] | 0.595 | 0.634 | 0.553 | 0.558 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| CNN + ECA, mean pool | 0.479 | 0.436 | 0.484 |
| CNN + ECA, attention pool | 0.512 | 0.452 | 0.480 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.553, 0.570, 0.575 | 0.566 ± 0.011 |
| CNN + ECA, attention pool | 0.540, 0.583, 0.616 | 0.580 ± 0.038 |

Median best epoch: CNN + ECA, mean pool 3, CNN + ECA, attention pool 13.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| CNN + ECA, mean pool | 0.661 [0.641, 0.680] | +0.017 [-0.006, +0.037] | 0.658 | 0.664 | 0.642 | 0.671 | 0.679 |
| CNN + ECA, attention pool | 0.672 [0.651, 0.691] | +0.028 [+0.005, +0.050] | 0.668 | 0.664 | 0.694 | 0.679 | 0.650 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| CNN + ECA, mean pool | 0.494 | 0.502 | 0.527 | 0.500 |
| CNN + ECA, attention pool | 0.481 | 0.508 | 0.542 | 0.526 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN + ECA, mean pool | 0.654, 0.670, 0.673 | 0.666 ± 0.010 |
| CNN + ECA, attention pool | 0.659, 0.678, 0.678 | 0.672 ± 0.011 |

Median best epoch: CNN + ECA, mean pool 17, CNN + ECA, attention pool 17.
