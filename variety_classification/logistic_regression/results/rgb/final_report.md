# Variety: logistic regression, RGB (3 bands)

Leaf mean spectrum → SNV → logistic regression, C chosen on the validation row. Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every metric: `metrics.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | — | 0.420 | 0.500 | 0.599 | 0.520 | 0.475 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.351 | 0.344 | 0.768 | 0.548 |

Chosen C per fold: 0.1, 1, 1000, 0.1, 0.1.
