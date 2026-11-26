# Variety: logistic regression, RGB (3 bands)

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression, C chosen on the validation row. Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every metric: `metrics.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.675 [0.649, 0.702] | — | 0.595 | 0.655 | 0.807 | 0.695 | 0.625 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.520 | 0.624 | 0.804 | 0.752 |

Chosen C per fold: 0.1, 1, 1000, 1, 1000.
