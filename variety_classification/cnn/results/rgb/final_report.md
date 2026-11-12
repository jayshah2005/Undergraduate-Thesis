# Variety: whole-leaf CNN, RGB (3 bands)

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input. Every metric: `metrics.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | — | 0.420 | 0.500 | 0.599 | 0.520 | 0.475 |
| CNN, mean pool | 0.824 [0.799, 0.846] | +0.321 [+0.286, +0.355] | 0.730 | 0.870 | 0.905 | 0.820 | 0.795 |
| small ResNet, mean pool | 0.885 [0.865, 0.904] | +0.382 [+0.349, +0.416] | 0.835 | 0.935 | 0.940 | 0.840 | 0.875 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.351 | 0.344 | 0.768 | 0.548 |
| CNN, mean pool | 0.794 | 0.880 | 0.924 | 0.696 |
| small ResNet, mean pool | 0.871 | 0.900 | 0.960 | 0.808 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.780, 0.791, 0.788 | 0.786 ± 0.005 |
| small ResNet, mean pool | 0.847, 0.859, 0.859 | 0.855 ± 0.007 |

Median best epoch: CNN, mean pool 49, small ResNet, mean pool 54.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, mean pool | +0.074 [+0.049, +0.100] |
| small ResNet, mean pool | +0.022 [-0.001, +0.046] |
| logistic regression | +0.371 [+0.341, +0.404] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
