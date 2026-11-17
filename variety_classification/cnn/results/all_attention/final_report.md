# Variety: whole-leaf CNN, Full spectrum (204 bands)

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input. Every metric: `metrics.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| CNN, attention pool | 0.895 [0.876, 0.914] | +0.021 [+0.005, +0.038] | 0.620 | 0.970 | 0.995 | 0.980 | 0.910 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.907 | 0.848 | 0.944 | 0.796 |
| CNN, attention pool | 0.931 | 0.876 | 0.976 | 0.796 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, attention pool | 0.887, 0.857, 0.884 | 0.876 ± 0.017 |

Median best epoch: 26.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, attention pool | +0.066 [+0.040, +0.092] |
| logistic regression | +0.371 [+0.341, +0.404] |

## Attention − mean pool, Full spectrum (204 bands) (same resampled leaves)

| backbone | Δ balanced accuracy [95% CI] |
|---|---|
| CNN | -0.003 [-0.018, +0.013] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
