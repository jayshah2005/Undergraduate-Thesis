# Variety: whole-leaf CNN, Full spectrum (204 bands)

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input. Every metric: `metrics.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| CNN, mean pool | 0.898 [0.880, 0.916] | +0.024 [+0.007, +0.041] | 0.595 | 0.975 | 0.975 | 0.975 | 0.970 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.907 | 0.848 | 0.944 | 0.796 |
| CNN, mean pool | 0.931 | 0.856 | 0.992 | 0.812 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.857, 0.856, 0.914 | 0.875 ± 0.033 |

Median best epoch: 11.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, mean pool | +0.074 [+0.049, +0.100] |
| logistic regression | +0.371 [+0.341, +0.404] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
