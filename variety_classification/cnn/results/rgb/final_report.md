# Variety: whole-leaf CNN, RGB (3 bands)

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input. Every metric: `metrics.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | — | 0.420 | 0.500 | 0.599 | 0.520 | 0.475 |
| CNN, mean pool | 0.824 [0.799, 0.846] | +0.321 [+0.286, +0.355] | 0.730 | 0.870 | 0.905 | 0.820 | 0.795 |
| CNN, attention pool | 0.829 [0.806, 0.851] | +0.326 [+0.294, +0.360] | 0.805 | 0.925 | 0.865 | 0.780 | 0.770 |
| small ResNet, mean pool | 0.885 [0.865, 0.904] | +0.382 [+0.349, +0.416] | 0.835 | 0.935 | 0.940 | 0.840 | 0.875 |
| small ResNet, attention pool | 0.834 [0.810, 0.856] | +0.331 [+0.297, +0.366] | 0.810 | 0.885 | 0.890 | 0.820 | 0.765 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.351 | 0.344 | 0.768 | 0.548 |
| CNN, mean pool | 0.794 | 0.880 | 0.924 | 0.696 |
| CNN, attention pool | 0.863 | 0.816 | 0.940 | 0.696 |
| small ResNet, mean pool | 0.871 | 0.900 | 0.960 | 0.808 |
| small ResNet, attention pool | 0.835 | 0.852 | 0.936 | 0.712 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.780, 0.791, 0.788 | 0.786 ± 0.005 |
| CNN, attention pool | 0.841, 0.786, 0.792 | 0.806 ± 0.030 |
| small ResNet, mean pool | 0.847, 0.859, 0.859 | 0.855 ± 0.007 |
| small ResNet, attention pool | 0.821, 0.811, 0.810 | 0.814 ± 0.006 |

Median best epoch: CNN, mean pool 49, CNN, attention pool 37, small ResNet, mean pool 54, small ResNet, attention pool 29.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, mean pool | +0.074 [+0.049, +0.100] |
| CNN, attention pool | +0.066 [+0.040, +0.092] |
| small ResNet, mean pool | +0.022 [-0.001, +0.046] |
| small ResNet, attention pool | +0.068 [+0.042, +0.095] |
| logistic regression | +0.371 [+0.341, +0.404] |

## Attention − mean pool, RGB (3 bands) (same resampled leaves)

| backbone | Δ balanced accuracy [95% CI] |
|---|---|
| CNN | +0.005 [-0.015, +0.027] |
| small ResNet | -0.051 [-0.069, -0.033] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
