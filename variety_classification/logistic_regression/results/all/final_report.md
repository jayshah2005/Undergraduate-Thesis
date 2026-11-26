# Variety: logistic regression, Full spectrum (204 bands)

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression, C chosen on the validation row. Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every metric: `metrics.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.885 [0.870, 0.900] | — | 0.635 | 0.975 | 0.980 | 0.940 | 0.895 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.915 | 0.836 | 0.960 | 0.828 |

Chosen C per fold: 0.01, 10, 10, 1000, 1.
