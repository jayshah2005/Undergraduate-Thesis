# Leaf spectra: variety vs disease - numbers

Specular glints: 20.60% of a leaf's pixels have a band above 1.0 on average (max 96.4%); they are included in the means.

## Share of the total variance (sum of squares, all bands) explained by

| grouping | raw | SNV |
|---|---|---|
| variety | 0.559 | 0.510 |
| class (pooled over varieties) | 0.108 | 0.051 |
| class within variety | 0.002 | 0.011 |
| row within variety | 0.083 | 0.083 |
| variety x row cells | 0.642 | 0.593 |

Class within variety and row within variety overlap (disease prevalence differs by row).

## PCA of all leaves (raw): first 10 PCs

| PC | explained | eta2 variety | eta2 class within variety | corr with brightness |
|---|---|---|---|---|
| 1 | 96.6% | 0.560 | 0.002 | +1.00 |
| 2 | 2.5% | 0.644 | 0.005 | +0.09 |
| 3 | 0.5% | 0.149 | 0.018 | -0.00 |
| 4 | 0.2% | 0.405 | 0.002 | -0.01 |
| 5 | 0.1% | 0.120 | 0.010 | +0.01 |
| 6 | 0.0% | 0.006 | 0.016 | +0.00 |
| 7 | 0.0% | 0.022 | 0.020 | +0.00 |
| 8 | 0.0% | 0.052 | 0.006 | -0.00 |
| 9 | 0.0% | 0.042 | 0.009 | +0.00 |
| 10 | 0.0% | 0.103 | 0.011 | -0.00 |

Silhouette in PC1-5 (1 = tight separate clusters, 0 = none): by variety 0.104, by class -0.076. Leave-one-row-out logistic regression predicts the variety with balanced accuracy 0.933 (chance 0.25).

## PCA of all leaves (SNV): first 10 PCs

| PC | explained | eta2 variety | eta2 class within variety | corr with brightness |
|---|---|---|---|---|
| 1 | 47.9% | 0.586 | 0.015 | +0.17 |
| 2 | 34.2% | 0.514 | 0.009 | +0.44 |
| 3 | 13.1% | 0.377 | 0.003 | -0.05 |
| 4 | 2.1% | 0.084 | 0.011 | -0.29 |
| 5 | 1.0% | 0.120 | 0.013 | -0.43 |
| 6 | 0.6% | 0.152 | 0.004 | -0.02 |
| 7 | 0.3% | 0.034 | 0.012 | +0.19 |
| 8 | 0.2% | 0.192 | 0.013 | -0.44 |
| 9 | 0.1% | 0.092 | 0.019 | +0.13 |
| 10 | 0.1% | 0.004 | 0.010 | -0.17 |

Silhouette in PC1-5 (1 = tight separate clusters, 0 = none): by variety 0.230, by class -0.015. Leave-one-row-out logistic regression predicts the variety with balanced accuracy 0.937 (chance 0.25).

## Baco: does disease separate the leaves?

R2 = share of this variety's spectral variance between the groups; chance = the R2 that shuffled labels give on average. p permutes the labels within rows (999 permutations). AUC = leave-one-row-out logistic regression, computed within rows (0.5 = chance).

| target | R2 raw | chance raw | p raw | AUC raw | R2 SNV | chance SNV | p SNV | AUC SNV |
|---|---|---|---|---|---|---|---|---|
| class (4) | 0.004 | 0.011 | 0.848 | - | 0.018 | 0.015 | 0.273 | - |
| healthy vs any disease | 0.001 | 0.004 | 0.659 | 0.473 | 0.012 | 0.008 | 0.223 | 0.516 |
| RB label | 0.002 | 0.004 | 0.510 | 0.520 | 0.012 | 0.005 | 0.084 | 0.482 |
| LR label | 0.001 | 0.004 | 0.714 | 0.506 | 0.002 | 0.005 | 0.750 | 0.484 |
| row (for scale; p from free permutation) | 0.086 | 0.016 | 0.001 | | 0.246 | 0.016 | 0.001 | |

PCA inside Baco (raw): eta2 of class / row per PC: PC1 0.00/0.08, PC2 0.02/0.15, PC3 0.01/0.22, PC4 0.01/0.38, PC5 0.00/0.08, PC6 0.03/0.06, PC7 0.00/0.06, PC8 0.02/0.21, PC9 0.01/0.03, PC10 0.01/0.05. Silhouette PC1-5 by class -0.039, by row -0.108.

## Chardonnay: does disease separate the leaves?

R2 = share of this variety's spectral variance between the groups; chance = the R2 that shuffled labels give on average. p permutes the labels within rows (999 permutations). AUC = leave-one-row-out logistic regression, computed within rows (0.5 = chance).

| target | R2 raw | chance raw | p raw | AUC raw | R2 SNV | chance SNV | p SNV | AUC SNV |
|---|---|---|---|---|---|---|---|---|
| class (4) | 0.007 | 0.013 | 0.640 | - | 0.075 | 0.015 | 0.001 | - |
| healthy vs any disease | 0.003 | 0.004 | 0.444 | 0.551 | 0.031 | 0.005 | 0.003 | 0.592 |
| RB label | 0.001 | 0.004 | 0.604 | 0.581 | 0.069 | 0.007 | 0.001 | 0.570 |
| LR label | 0.006 | 0.004 | 0.261 | 0.527 | 0.009 | 0.004 | 0.117 | 0.462 |
| row (for scale; p from free permutation) | 0.056 | 0.016 | 0.007 | | 0.063 | 0.016 | 0.001 | |

PCA inside Chardonnay (raw): eta2 of class / row per PC: PC1 0.01/0.06, PC2 0.06/0.09, PC3 0.04/0.02, PC4 0.03/0.03, PC5 0.01/0.04, PC6 0.01/0.04, PC7 0.01/0.14, PC8 0.01/0.07, PC9 0.03/0.07, PC10 0.02/0.03. Silhouette PC1-5 by class -0.044, by row -0.103.

## Largest per-band effect sizes (Cohen's d, raw spectra)

| comparison | max abs d | at nm |
|---|---|---|
| Baco: healthy vs RB only | 0.22 | 403 |
| Baco: healthy vs LR only | 0.16 | 403 |
| Baco: healthy vs any disease | 0.18 | 403 |
| Chardonnay: healthy vs RB only | 0.38 | 707 |
| Chardonnay: healthy vs LR only | 0.24 | 707 |
| Chardonnay: healthy vs any disease | 0.36 | 707 |
| variety: Baco vs Chardonnay | 4.51 | 400 |
| variety: Baco vs Vidal | 4.83 | 400 |
| variety: Chardonnay vs Vidal | 2.05 | 787 |
| variety: Cab Franc vs Vidal | 0.73 | 554 |
