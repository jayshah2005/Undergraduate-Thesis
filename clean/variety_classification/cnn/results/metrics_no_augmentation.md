# Variety: whole-leaf CNN, trained without augmentation

Balanced accuracy = the mean of the 4 recalls (chance 0.25), with a 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on the validation row: the CNN its checkpoint, logistic regression its C. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression on the same input.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| CNN, mean pool | 0.914 [0.898, 0.931] | +0.040 [+0.023, +0.058] | 0.670 | 0.960 | 0.995 | 0.995 | 0.950 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.907 | 0.848 | 0.944 | 0.796 |
| CNN, mean pool | 0.972 | 0.788 | 0.984 | 0.912 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.889, 0.883, 0.854 | 0.875 ± 0.019 |

Median best epoch: 13.

## RGB (3 bands)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | — | 0.420 | 0.500 | 0.599 | 0.520 | 0.475 |
| CNN, mean pool | 0.928 [0.911, 0.943] | +0.425 [+0.395, +0.457] | 0.845 | 0.970 | 0.995 | 0.965 | 0.865 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Recall per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.351 | 0.344 | 0.768 | 0.548 |
| CNN, mean pool | 0.907 | 0.908 | 0.980 | 0.916 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.916, 0.931, 0.896 | 0.914 ± 0.018 |

Median best epoch: 70.

## 204 bands − RGB (same resampled leaves)

| model | Δ balanced accuracy [95% CI] |
|---|---|
| CNN, mean pool | -0.014 [-0.034, +0.006] |
| logistic regression | +0.371 [+0.341, +0.404] |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.
