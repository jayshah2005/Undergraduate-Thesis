# Variety: whole-leaf CNN, RGB (3 bands)

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input. Every metric: `metrics.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | — | 0.420 | 0.500 | 0.599 | 0.520 | 0.475 |
| CNN, attention pool | 0.829 [0.806, 0.851] | +0.326 [+0.294, +0.360] | 0.805 | 0.925 | 0.865 | 0.780 | 0.770 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.351 | 0.344 | 0.768 | 0.548 |
| CNN, attention pool | 0.863 | 0.816 | 0.940 | 0.696 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, attention pool | 0.841, 0.786, 0.792 | 0.806 ± 0.030 |

Median best epoch: 37.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, attention pool | +0.066 [+0.040, +0.092] |
| logistic regression | +0.371 [+0.341, +0.404] |

## Attention − mean pool, RGB (3 bands) (same resampled leaves)

| backbone | Δ balanced accuracy [95% CI] |
|---|---|
| CNN | +0.005 [-0.015, +0.027] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
