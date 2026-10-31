# Is the spectral signature different by disease class?

SNV-normalised leaf-mean spectra (brightness removed). Classes: healthy / RB only / LR only / both. R² is the share of spectral variance between those groups. Disease p-values permute labels *within vineyard rows*. Pooled class R² is inflated by variety (Cab Franc ≈ 96% RB only).

## Share of total variance (SNV)

| grouping | R² |
|---|---|
| variety | 0.510 |
| class (pooled over varieties) | 0.051 |
| class within variety | 0.011 |
| row within variety | 0.083 |

## Baco: does disease separate the vines?

| target | R² | chance | p |
|---|---|---|---|
| class (4) | 0.018 | 0.014 | 0.268 |
| healthy vs any disease | 0.012 | 0.009 | 0.242 |
| RB label | 0.012 | 0.005 | 0.085 |
| LR label | 0.002 | 0.005 | 0.771 |
| row (for scale; free permutation) | 0.246 | 0.016 | 0.001 |

PCA eta² of class / row per PC: PC1 0.02/0.26, PC2 0.00/0.20, PC3 0.01/0.37, PC4 0.02/0.06, PC5 0.00/0.05, PC6 0.01/0.16, PC7 0.01/0.04, PC8 0.01/0.06, PC9 0.01/0.01, PC10 0.04/0.02.

## Cab Franc: does disease separate the vines?

| target | R² | chance | p |
|---|---|---|---|
| class (4) | 0.002 | 0.008 | 0.914 |
| healthy vs any disease | 0.001 | 0.004 | 0.764 |
| RB label | 0.001 | 0.004 | 0.797 |
| LR label | 0.001 | 0.004 | 0.832 |
| row (for scale; free permutation) | 0.122 | 0.016 | 0.001 |

PCA eta² of class / row per PC: PC1 0.00/0.12, PC2 0.00/0.04, PC3 0.00/0.09, PC4 0.01/0.76, PC5 0.02/0.07, PC6 0.01/0.02, PC7 0.00/0.03, PC8 0.02/0.12, PC9 0.01/0.09, PC10 0.01/0.04.

## Chardonnay: does disease separate the vines?

| target | R² | chance | p |
|---|---|---|---|
| class (4) | 0.075 | 0.014 | 0.001 |
| healthy vs any disease | 0.031 | 0.005 | 0.005 |
| RB label | 0.069 | 0.006 | 0.001 |
| LR label | 0.009 | 0.005 | 0.141 |
| row (for scale; free permutation) | 0.063 | 0.016 | 0.001 |

PCA eta² of class / row per PC: PC1 0.10/0.08, PC2 0.00/0.02, PC3 0.01/0.02, PC4 0.01/0.01, PC5 0.01/0.09, PC6 0.01/0.06, PC7 0.04/0.14, PC8 0.02/0.01, PC9 0.04/0.06, PC10 0.02/0.06.

## Vidal: does disease separate the vines?

| target | R² | chance | p |
|---|---|---|---|
| class (4) | 0.026 | 0.011 | 0.037 |
| healthy vs any disease | 0.023 | 0.003 | 0.003 |
| RB label | 0.020 | 0.003 | 0.004 |
| LR label | 0.002 | 0.004 | 0.699 |
| row (for scale; free permutation) | 0.216 | 0.016 | 0.001 |

PCA eta² of class / row per PC: PC1 0.03/0.12, PC2 0.01/0.65, PC3 0.00/0.10, PC4 0.03/0.05, PC5 0.02/0.01, PC6 0.01/0.01, PC7 0.02/0.09, PC8 0.01/0.01, PC9 0.01/0.10, PC10 0.01/0.10.

## Largest per-band effect sizes (Cohen's d, SNV)

| comparison | max abs d | at nm |
|---|---|---|
| Baco: healthy vs RB only | 0.46 | 607 |
| Baco: healthy vs LR only | 0.36 | 973 |
| Baco: healthy vs both | 0.38 | 973 |
| Baco: healthy vs any disease | 0.32 | 697 |
| Cab Franc: healthy vs RB only | 0.84 | 936 |
| Cab Franc: healthy vs any disease | 0.84 | 936 |
| Chardonnay: healthy vs RB only | 0.81 | 703 |
| Chardonnay: healthy vs LR only | 0.24 | 694 |
| Chardonnay: healthy vs both | 0.84 | 703 |
| Chardonnay: healthy vs any disease | 0.49 | 700 |
| Vidal: healthy vs RB only | 0.49 | 537 |
| Vidal: healthy vs LR only | 0.72 | 967 |
| Vidal: healthy vs both | 0.71 | 640 |
| Vidal: healthy vs any disease | 0.49 | 534 |
