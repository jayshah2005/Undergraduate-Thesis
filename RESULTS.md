# Results

Built by `make_results.py` from the saved predictions; the plan, fixed before the runs, is [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).

**How to read it.** Each score is the seed ensemble (the mean probability of 5 seeds; logistic regression and the variety-only rule are one fit) on the held-out rows, with a 95% interval from 2000 resamples of leaves within each row × class (× variety when pooled). Balanced accuracy: chance 0.5 for a disease, 0.25 for variety. AUC: chance 0.5. Each comparison A − B is paired (the same resamples for both models). It gets ✓ (A better) or ✗ (A worse) only when its 95% interval excludes 0 **and** A beats (or loses to) B in at least 4 of the 5 seeds trained separately (seed s against seed s; a CNN seed against the one logistic-regression fit). Two single fits (logistic regression 204 bands − RGB) have only the interval and the sign. Unmarked rows are ties within the noise.

**Not finished yet (76 models; left out of every table and count):** CNN + CBAM, attention pool (RGB); CNN + CBAM, mean pool (RGB); CNN + ECA, attention pool (RGB); CNN + ECA, mean pool (RGB); CNN, attention pool (RGB); CNN, mean pool (RGB); six-head CNN, AdamW, LS (204 bands); six-head CNN, AdamW, LS (RGB); six-head CNN, AdamW, MGDA (204 bands); six-head CNN, AdamW, MGDA (RGB); six-head CNN, AdamW, MGDA-GN (204 bands); six-head CNN, AdamW, MGDA-GN (RGB); six-head CNN, AdamW, MGDA-LGN (204 bands); six-head CNN, AdamW, MGDA-LGN (RGB); six-head CNN, AdamW, MGDA-LN (204 bands); six-head CNN, AdamW, MGDA-LN (RGB); six-head CNN, AdamW, OSNMF-PGA (204 bands); six-head CNN, AdamW, OSNMF-PGA (RGB); six-head CNN, AdamW, SVD-PGA (204 bands); six-head CNN, AdamW, SVD-PGA (RGB); six-head CNN, AdamW, UPGrad (204 bands); six-head CNN, AdamW, UPGrad (RGB); six-head CNN, SGD, LS (204 bands); six-head CNN, SGD, LS (RGB); six-head CNN, SGD, MGDA (204 bands); six-head CNN, SGD, MGDA (RGB); six-head CNN, SGD, MGDA-GN (204 bands); six-head CNN, SGD, MGDA-GN (RGB); six-head CNN, SGD, MGDA-LGN (204 bands); six-head CNN, SGD, MGDA-LGN (RGB); six-head CNN, SGD, MGDA-LN (204 bands); six-head CNN, SGD, MGDA-LN (RGB); six-head CNN, SGD, OSNMF-PGA (204 bands); six-head CNN, SGD, OSNMF-PGA (RGB); six-head CNN, SGD, SVD-PGA (204 bands); six-head CNN, SGD, SVD-PGA (RGB); six-head CNN, SGD, UPGrad (204 bands); six-head CNN, SGD, UPGrad (RGB); six-head small ResNet, AdamW, LS (204 bands); six-head small ResNet, AdamW, LS (RGB); six-head small ResNet, AdamW, MGDA (204 bands); six-head small ResNet, AdamW, MGDA (RGB); six-head small ResNet, AdamW, MGDA-GN (204 bands); six-head small ResNet, AdamW, MGDA-GN (RGB); six-head small ResNet, AdamW, MGDA-LGN (204 bands); six-head small ResNet, AdamW, MGDA-LGN (RGB); six-head small ResNet, AdamW, MGDA-LN (204 bands); six-head small ResNet, AdamW, MGDA-LN (RGB); six-head small ResNet, AdamW, OSNMF-PGA (204 bands); six-head small ResNet, AdamW, OSNMF-PGA (RGB); six-head small ResNet, AdamW, SVD-PGA (204 bands); six-head small ResNet, AdamW, SVD-PGA (RGB); six-head small ResNet, AdamW, UPGrad (204 bands); six-head small ResNet, AdamW, UPGrad (RGB); six-head small ResNet, SGD, LS (204 bands); six-head small ResNet, SGD, LS (RGB); six-head small ResNet, SGD, MGDA (204 bands); six-head small ResNet, SGD, MGDA (RGB); six-head small ResNet, SGD, MGDA-GN (204 bands); six-head small ResNet, SGD, MGDA-GN (RGB); six-head small ResNet, SGD, MGDA-LGN (204 bands); six-head small ResNet, SGD, MGDA-LGN (RGB); six-head small ResNet, SGD, MGDA-LN (204 bands); six-head small ResNet, SGD, MGDA-LN (RGB); six-head small ResNet, SGD, OSNMF-PGA (204 bands); six-head small ResNet, SGD, OSNMF-PGA (RGB); six-head small ResNet, SGD, SVD-PGA (204 bands); six-head small ResNet, SGD, SVD-PGA (RGB); six-head small ResNet, SGD, UPGrad (204 bands); six-head small ResNet, SGD, UPGrad (RGB); small ResNet + CBAM, attention pool (RGB); small ResNet + CBAM, mean pool (RGB); small ResNet + ECA, attention pool (RGB); small ResNet + ECA, mean pool (RGB); small ResNet, attention pool (RGB); small ResNet, mean pool (RGB).

## Contents

1. [Summary of the marks](#summary-of-the-marks)
2. [Variety](#variety)
3. [Disease](#disease): [Leafroll](#leafroll-overview), [Red blotch](#red-blotch-overview)
4. [MOO (six heads)](#moo-six-heads)

## Summary of the marks

### Variety

| # | comparison | rows | chance (1 in 20) | bal. acc. ✓ | bal. acc. ✗ | AUC ✓ | AUC ✗ |
|---|---|---|---|---|---|---|---|
| 1 | against logistic regression, same data | 13 | 0.7 | 3 | 0 | 1 | 1 |
| 2a | ECA / CBAM against the same network and pooling without them | 8 | 0.4 | 2 | 1 | 4 | 1 |
| 2b | attention against mean pooling, same network | 6 | 0.3 | 0 | 1 | 4 | 0 |
| 3 | small ResNet against CNN, same module and pooling | 6 | 0.3 | 2 | 1 | 1 | 2 |
| 4 | 204 bands against RGB, same model | 2 | 0.1 | 1 | 0 | 1 | 0 |

### Disease (every setting)

| # | comparison | rows | chance (1 in 20) | bal. acc. ✓ | bal. acc. ✗ | AUC ✓ | AUC ✗ |
|---|---|---|---|---|---|---|---|
| 1 | against logistic regression, same data | 120 | 6.0 | 24 | 5 | 24 | 1 |
| 1b | against the variety-only rule (pooled sets) | 56 | 2.8 | 0 | 26 | 0 | 13 |
| 2a | ECA / CBAM against the same network and pooling without them | 80 | 4.0 | 9 | 3 | 14 | 7 |
| 2b | attention against mean pooling, same network | 60 | 3.0 | 1 | 7 | 3 | 8 |
| 3 | small ResNet against CNN, same module and pooling | 60 | 3.0 | 3 | 4 | 9 | 6 |
| 4 | 204 bands against RGB, same model | 10 | 0.5 | 0 | 1 | 1 | 1 |

### MOO

| # | comparison | rows | chance (1 in 20) | bal. acc. ✓ | bal. acc. ✗ | AUC ✓ | AUC ✗ |
|---|---|---|---|---|---|---|---|

### Every marked comparison

Variety:

| setting | # | A − B | bands | Δ bal. acc. [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|---|
| Variety (4 classes), all 998 leaves | 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | +0.040 [+0.025, +0.055] ✓ | +0.004 [+0.001, +0.008]  |
| Variety (4 classes), all 998 leaves | 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | +0.024 [+0.009, +0.040] ✓ | +0.006 [+0.003, +0.010]  |
| Variety (4 classes), all 998 leaves | 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | +0.027 [+0.013, +0.042]  | -0.006 [-0.010, -0.002] ✗ |
| Variety (4 classes), all 998 leaves | 1 | CNN, mean pool − logistic regression | RGB | +0.243 [+0.216, +0.270] ✓ | +0.125 [+0.110, +0.139] ✓ |
| Variety (4 classes), all 998 leaves | 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | -0.017 [-0.028, -0.007] ✗ | -0.005 [-0.006, -0.003] ✗ |
| Variety (4 classes), all 998 leaves | 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | +0.020 [+0.008, +0.033]  | +0.003 [+0.002, +0.005] ✓ |
| Variety (4 classes), all 998 leaves | 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | +0.012 [+0.003, +0.022] ✓ | +0.002 [+0.000, +0.003]  |
| Variety (4 classes), all 998 leaves | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | +0.033 [+0.022, +0.045]  | +0.003 [+0.000, +0.005] ✓ |
| Variety (4 classes), all 998 leaves | 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | -0.002 [-0.011, +0.008]  | +0.007 [+0.004, +0.009] ✓ |
| Variety (4 classes), all 998 leaves | 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | +0.014 [+0.005, +0.023] ✓ | +0.006 [+0.004, +0.008] ✓ |
| Variety (4 classes), all 998 leaves | 2b | CNN, attention pool − CNN, mean pool | 204 bands | -0.028 [-0.040, -0.016] ✗ | -0.001 [-0.003, +0.000]  |
| Variety (4 classes), all 998 leaves | 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | +0.009 [-0.004, +0.022]  | +0.007 [+0.005, +0.008] ✓ |
| Variety (4 classes), all 998 leaves | 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | -0.014 [-0.024, -0.004]  | +0.003 [+0.000, +0.006] ✓ |
| Variety (4 classes), all 998 leaves | 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | +0.002 [-0.009, +0.012]  | +0.003 [+0.001, +0.004] ✓ |
| Variety (4 classes), all 998 leaves | 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | -0.013 [-0.024, -0.002]  | +0.006 [+0.003, +0.008] ✓ |
| Variety (4 classes), all 998 leaves | 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | -0.004 [-0.013, +0.004]  | -0.006 [-0.008, -0.004] ✗ |
| Variety (4 classes), all 998 leaves | 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | +0.010 [+0.000, +0.020] ✓ | -0.001 [-0.003, +0.001]  |
| Variety (4 classes), all 998 leaves | 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | +0.011 [+0.000, +0.022] ✓ | +0.005 [+0.004, +0.007] ✓ |
| Variety (4 classes), all 998 leaves | 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | -0.013 [-0.023, -0.003]  | -0.010 [-0.013, -0.008] ✗ |
| Variety (4 classes), all 998 leaves | 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | -0.019 [-0.030, -0.009] ✗ | -0.004 [-0.006, -0.002]  |
| Variety (4 classes), all 998 leaves | 4 | logistic regression − logistic regression | 204 bands − RGB | +0.210 [+0.183, +0.237] ✓ | +0.121 [+0.107, +0.135] ✓ |

Disease:

| setting | # | A − B | bands | Δ bal. acc. [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|---|
| Leafroll, Baco | 1 | CNN + ECA, mean pool − logistic regression | 204 bands | +0.153 [+0.096, +0.209] ✓ | +0.085 [+0.016, +0.149] ✓ |
| Leafroll, Baco | 1 | CNN + ECA, attention pool − logistic regression | 204 bands | +0.135 [+0.069, +0.200] ✓ | +0.091 [+0.023, +0.155] ✓ |
| Leafroll, Baco | 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | +0.157 [+0.087, +0.226] ✓ | +0.106 [+0.039, +0.170] ✓ |
| Leafroll, Baco | 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | +0.117 [+0.044, +0.188] ✓ | +0.131 [+0.061, +0.200] ✓ |
| Leafroll, Baco | 1 | small ResNet, mean pool − logistic regression | 204 bands | +0.113 [+0.049, +0.174] ✓ | +0.093 [+0.028, +0.154] ✓ |
| Leafroll, Baco | 1 | small ResNet, attention pool − logistic regression | 204 bands | +0.148 [+0.086, +0.208] ✓ | +0.100 [+0.038, +0.159] ✓ |
| Leafroll, Baco | 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | +0.122 [+0.049, +0.193] ✓ | +0.082 [+0.011, +0.149] ✓ |
| Leafroll, Baco | 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | +0.127 [+0.068, +0.182] ✓ | +0.096 [+0.031, +0.156] ✓ |
| Leafroll, Baco | 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | +0.115 [+0.060, +0.168] ✓ | +0.106 [+0.045, +0.165] ✓ |
| Leafroll, Baco | 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | +0.107 [+0.044, +0.173] ✓ | +0.076 [+0.011, +0.138] ✓ |
| Leafroll, Baco | 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | +0.093 [+0.060, +0.126]  | +0.031 [+0.008, +0.054] ✓ |
| Leafroll, Baco | 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | +0.097 [+0.064, +0.133] ✓ | +0.030 [+0.009, +0.050] ✓ |
| Leafroll, Baco | 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | +0.097 [+0.059, +0.133] ✓ | +0.052 [+0.026, +0.079] ✓ |
| Leafroll, Baco | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | +0.079 [+0.030, +0.130]  | +0.070 [+0.039, +0.101] ✓ |
| Leafroll, Baco | 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | -0.039 [-0.077, -0.000] ✗ | +0.025 [-0.001, +0.052]  |
| Leafroll, Baco | 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | -0.008 [-0.048, +0.030]  | -0.031 [-0.057, -0.002] ✗ |
| Leafroll, Baco | 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | +0.053 [+0.015, +0.089] ✓ | +0.039 [+0.019, +0.061] ✓ |
| Leafroll, Baco | 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | +0.110 [+0.068, +0.151]  | +0.038 [+0.018, +0.058] ✓ |
| Leafroll, Baco | 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | -0.041 [-0.066, -0.016] ✗ | +0.001 [-0.024, +0.024]  |
| Leafroll, Baco | 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | -0.010 [-0.058, +0.034]  | -0.055 [-0.086, -0.024] ✗ |
| Leafroll, Chardonnay | 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | +0.026 [-0.002, +0.058]  | -0.023 [-0.045, -0.000] ✗ |
| Leafroll, Vidal | 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | -0.074 [-0.135, -0.027] ✗ | -0.012 [-0.073, +0.036]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | +0.039 [+0.008, +0.070] ✓ | +0.048 [+0.024, +0.070] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1 | small ResNet, attention pool − logistic regression | 204 bands | +0.024 [-0.007, +0.057]  | +0.042 [+0.019, +0.064] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | logistic regression − variety-only rule | 204 bands | -0.089 [-0.120, -0.061] ✗ | -0.033 [-0.058, -0.008] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN, mean pool − variety-only rule | 204 bands | -0.079 [-0.098, -0.060] ✗ | -0.016 [-0.035, +0.002]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN, attention pool − variety-only rule | 204 bands | -0.081 [-0.100, -0.060] ✗ | -0.028 [-0.049, -0.008] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | -0.071 [-0.090, -0.050] ✗ | -0.016 [-0.036, +0.004]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | -0.069 [-0.087, -0.051] ✗ | -0.019 [-0.037, -0.001] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN + CBAM, mean pool − variety-only rule | 204 bands | -0.065 [-0.085, -0.045] ✗ | -0.001 [-0.022, +0.019]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | -0.051 [-0.072, -0.029] ✗ | +0.014 [-0.004, +0.032]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet, mean pool − variety-only rule | 204 bands | -0.092 [-0.114, -0.072] ✗ | -0.020 [-0.040, +0.000]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet, attention pool − variety-only rule | 204 bands | -0.065 [-0.087, -0.041] ✗ | +0.009 [-0.012, +0.030]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet + ECA, mean pool − variety-only rule | 204 bands | -0.072 [-0.092, -0.053] ✗ | +0.003 [-0.016, +0.022]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | -0.103 [-0.122, -0.084] ✗ | -0.030 [-0.049, -0.011] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet + CBAM, mean pool − variety-only rule | 204 bands | -0.064 [-0.082, -0.047] ✗ | -0.015 [-0.031, +0.002]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | small ResNet + CBAM, attention pool − variety-only rule | 204 bands | -0.092 [-0.110, -0.074] ✗ | -0.026 [-0.044, -0.007] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 1b | logistic regression − variety-only rule | RGB | -0.103 [-0.130, -0.077] ✗ | -0.030 [-0.057, -0.005] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | +0.030 [+0.007, +0.053] ✓ | +0.043 [+0.026, +0.060] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | -0.038 [-0.059, -0.018] ✗ | -0.039 [-0.051, -0.026] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | -0.027 [-0.051, -0.005] ✗ | -0.035 [-0.049, -0.020] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | +0.027 [+0.002, +0.054]  | +0.029 [+0.010, +0.049] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | -0.031 [-0.053, -0.011]  | -0.032 [-0.051, -0.014] ✗ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | -0.028 [-0.049, -0.007] ✗ | -0.011 [-0.025, +0.003]  |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | +0.016 [-0.005, +0.038]  | +0.037 [+0.020, +0.054] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | -0.001 [-0.022, +0.019]  | +0.019 [+0.004, +0.033] ✓ |
| Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc) | 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | -0.041 [-0.066, -0.018] ✗ | -0.040 [-0.056, -0.025] ✗ |
| Red blotch, Baco | 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | -0.077 [-0.141, -0.008] ✗ | -0.055 [-0.118, +0.014]  |
| Red blotch, Chardonnay | 1 | CNN, mean pool − logistic regression | 204 bands | -0.074 [-0.125, -0.022] ✗ | -0.026 [-0.065, +0.012]  |
| Red blotch, Chardonnay | 1 | CNN, attention pool − logistic regression | 204 bands | -0.084 [-0.133, -0.033] ✗ | -0.030 [-0.068, +0.008]  |
| Red blotch, Chardonnay | 1 | CNN + ECA, mean pool − logistic regression | 204 bands | -0.051 [-0.097, -0.001] ✗ | +0.020 [-0.021, +0.060]  |
| Red blotch, Chardonnay | 1 | small ResNet, attention pool − logistic regression | 204 bands | -0.055 [-0.100, -0.006] ✗ | -0.023 [-0.067, +0.022]  |
| Red blotch, Chardonnay | 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | +0.023 [-0.015, +0.061]  | +0.046 [+0.025, +0.066] ✓ |
| Red blotch, Chardonnay | 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | +0.060 [+0.017, +0.102] ✓ | +0.003 [-0.021, +0.025]  |
| Red blotch, Chardonnay | 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | +0.037 [-0.009, +0.083]  | +0.037 [+0.002, +0.070] ✓ |
| Red blotch, Chardonnay | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | +0.079 [+0.031, +0.127] ✓ | +0.048 [+0.014, +0.080] ✓ |
| Red blotch, Chardonnay | 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | +0.026 [-0.021, +0.070]  | -0.047 [-0.067, -0.028] ✗ |
| Red blotch, Chardonnay | 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | +0.066 [+0.025, +0.110]  | +0.030 [+0.008, +0.054] ✓ |
| Red blotch, Chardonnay | 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | +0.044 [-0.000, +0.085]  | -0.027 [-0.049, -0.006] ✗ |
| Red blotch, Chardonnay | 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | +0.023 [-0.014, +0.060]  | +0.024 [+0.004, +0.045] ✓ |
| Red blotch, Chardonnay | 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | -0.037 [-0.080, +0.008]  | -0.043 [-0.073, -0.013] ✗ |
| Red blotch, Chardonnay | 4 | logistic regression − logistic regression | 204 bands − RGB | -0.037 [-0.091, +0.016]  | -0.090 [-0.149, -0.030] ✗ |
| Red blotch, Vidal | 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | -0.028 [-0.092, +0.034]  | -0.077 [-0.135, -0.021] ✗ |
| Red blotch, Vidal | 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | +0.015 [-0.028, +0.058]  | -0.040 [-0.076, -0.004] ✗ |
| Red blotch, Vidal | 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | -0.069 [-0.117, -0.022] ✗ | -0.055 [-0.084, -0.026] ✗ |
| Red blotch, Vidal | 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | -0.051 [-0.095, -0.003]  | -0.048 [-0.079, -0.018] ✗ |
| Red blotch, Vidal | 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | -0.061 [-0.100, -0.024] ✗ | -0.063 [-0.094, -0.035] ✗ |
| Red blotch, Vidal | 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | -0.018 [-0.059, +0.025]  | -0.055 [-0.084, -0.024] ✗ |
| Red blotch, Vidal | 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | +0.075 [+0.032, +0.122] ✓ | +0.029 [-0.002, +0.058]  |
| Red blotch, Vidal | 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | -0.021 [-0.066, +0.022]  | -0.030 [-0.060, -0.002] ✗ |
| Red blotch, pooled, all 998 leaves | 1 | CNN, mean pool − logistic regression | 204 bands | +0.030 [+0.007, +0.052] ✓ | +0.025 [+0.011, +0.039] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | CNN, attention pool − logistic regression | 204 bands | +0.014 [-0.009, +0.036]  | +0.024 [+0.009, +0.039] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | +0.039 [+0.016, +0.062] ✓ | +0.040 [+0.024, +0.056] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | +0.036 [+0.013, +0.057] ✓ | +0.039 [+0.025, +0.055] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | small ResNet, mean pool − logistic regression | 204 bands | +0.028 [+0.005, +0.051] ✓ | +0.031 [+0.018, +0.045] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | +0.027 [+0.003, +0.051] ✓ | +0.031 [+0.017, +0.046] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | +0.005 [-0.017, +0.027]  | +0.024 [+0.009, +0.037] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | +0.026 [+0.003, +0.048] ✓ | +0.031 [+0.016, +0.046] ✓ |
| Red blotch, pooled, all 998 leaves | 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | +0.026 [+0.001, +0.049] ✓ | +0.037 [+0.022, +0.051] ✓ |
| Red blotch, pooled, all 998 leaves | 1b | logistic regression − variety-only rule | 204 bands | -0.043 [-0.063, -0.022] ✗ | -0.041 [-0.059, -0.022] ✗ |
| Red blotch, pooled, all 998 leaves | 1b | CNN, attention pool − variety-only rule | 204 bands | -0.028 [-0.046, -0.012] ✗ | -0.016 [-0.032, -0.001] ✗ |
| Red blotch, pooled, all 998 leaves | 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | -0.021 [-0.036, -0.005] ✗ | -0.012 [-0.027, +0.002]  |
| Red blotch, pooled, all 998 leaves | 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | -0.023 [-0.041, -0.005] ✗ | -0.023 [-0.038, -0.008] ✗ |
| Red blotch, pooled, all 998 leaves | 1b | small ResNet, attention pool − variety-only rule | 204 bands | -0.020 [-0.039, -0.002] ✗ | -0.012 [-0.027, +0.002]  |
| Red blotch, pooled, all 998 leaves | 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | -0.037 [-0.055, -0.019] ✗ | -0.017 [-0.033, -0.002] ✗ |
| Red blotch, pooled, all 998 leaves | 1b | logistic regression − variety-only rule | RGB | -0.065 [-0.087, -0.042] ✗ | -0.081 [-0.104, -0.058] ✗ |
| Red blotch, pooled, all 998 leaves | 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | +0.009 [-0.008, +0.027]  | +0.015 [+0.005, +0.025] ✓ |
| Red blotch, pooled, all 998 leaves | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | +0.021 [+0.004, +0.040] ✓ | +0.015 [+0.005, +0.026] ✓ |
| Red blotch, pooled, all 998 leaves | 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | +0.003 [-0.015, +0.021]  | +0.008 [+0.001, +0.016] ✓ |
| Red blotch, pooled, all 998 leaves | 2b | CNN, attention pool − CNN, mean pool | 204 bands | -0.015 [-0.029, -0.002] ✗ | -0.000 [-0.008, +0.007]  |
| Red blotch, pooled, all 998 leaves | 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | -0.022 [-0.039, -0.006] ✗ | -0.007 [-0.015, +0.001]  |
| Red blotch, pooled, all 998 leaves | 4 | logistic regression − logistic regression | 204 bands − RGB | +0.022 [-0.004, +0.049]  | +0.041 [+0.019, +0.064] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | CNN, attention pool − logistic regression | 204 bands | +0.029 [+0.004, +0.051] ✓ | +0.011 [-0.005, +0.027]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | CNN + ECA, attention pool − logistic regression | 204 bands | +0.045 [+0.020, +0.068] ✓ | +0.009 [-0.008, +0.026]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | +0.036 [+0.008, +0.063] ✓ | +0.014 [-0.005, +0.033]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | +0.035 [+0.007, +0.062] ✓ | +0.014 [-0.005, +0.032]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | +0.029 [+0.003, +0.055] ✓ | +0.010 [-0.008, +0.029]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | +0.037 [+0.009, +0.064] ✓ | +0.013 [-0.006, +0.030]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1b | logistic regression − variety-only rule | 204 bands | -0.039 [-0.062, -0.014] ✗ | -0.012 [-0.035, +0.013]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | -0.023 [-0.045, +0.001]  | -0.027 [-0.048, -0.007] ✗ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | -0.029 [-0.052, -0.007] ✗ | -0.017 [-0.039, +0.005]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 1b | small ResNet, attention pool − variety-only rule | 204 bands | -0.032 [-0.055, -0.010] ✗ | -0.009 [-0.029, +0.012]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | -0.008 [-0.027, +0.012]  | -0.019 [-0.028, -0.011] ✗ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | -0.019 [-0.037, +0.001]  | -0.016 [-0.026, -0.007] ✗ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | +0.028 [+0.010, +0.048] ✓ | +0.010 [+0.002, +0.019] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | +0.030 [+0.012, +0.050] ✓ | +0.010 [+0.002, +0.017]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | +0.028 [+0.008, +0.048] ✓ | +0.025 [+0.017, +0.034] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | -0.021 [-0.042, -0.001] ✗ | -0.008 [-0.019, +0.002]  |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | +0.020 [-0.002, +0.041]  | +0.030 [+0.017, +0.042] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | +0.001 [-0.019, +0.022]  | +0.013 [+0.004, +0.023] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | +0.027 [+0.005, +0.049] ✓ | +0.018 [+0.009, +0.028] ✓ |
| Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal) | 4 | logistic regression − logistic regression | 204 bands − RGB | -0.030 [-0.058, -0.001] ✗ | -0.013 [-0.037, +0.012]  |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | +0.025 [-0.002, +0.053]  | +0.030 [+0.009, +0.051] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1 | small ResNet, mean pool − logistic regression | 204 bands | +0.010 [-0.016, +0.036]  | +0.026 [+0.008, +0.044] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | +0.022 [-0.006, +0.050]  | +0.029 [+0.009, +0.049] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1b | CNN, attention pool − variety-only rule | 204 bands | -0.023 [-0.042, -0.003] ✗ | +0.007 [-0.013, +0.028]  |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | -0.028 [-0.050, -0.005] ✗ | -0.000 [-0.021, +0.021]  |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 1b | logistic regression − variety-only rule | RGB | -0.024 [-0.050, +0.002]  | -0.031 [-0.058, -0.004] ✗ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | +0.012 [-0.007, +0.030]  | -0.013 [-0.024, -0.003] ✗ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | +0.009 [-0.010, +0.030]  | +0.015 [+0.002, +0.028] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | -0.008 [-0.026, +0.009]  | -0.011 [-0.022, -0.001] ✗ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | +0.021 [+0.001, +0.042] ✓ | +0.012 [+0.001, +0.022] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2b | CNN, attention pool − CNN, mean pool | 204 bands | -0.020 [-0.036, -0.004] ✗ | +0.003 [-0.007, +0.014]  |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | +0.020 [-0.001, +0.041]  | +0.014 [+0.005, +0.024] ✓ |
| Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998) | 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | -0.023 [-0.047, -0.002] ✗ | -0.015 [-0.029, -0.002] ✗ |

MOO:

No comparison is marked.

## Variety

### Variety (4 classes), all 998 leaves

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.885 [0.870, 0.900] | 0.985 [0.981, 0.988] | 0.675 [0.649, 0.702] | 0.864 [0.848, 0.880] |
| CNN, mean pool | 0.913 [0.903, 0.922] | 0.987 [0.985, 0.989] | 0.918 [0.903, 0.932] | 0.988 [0.985, 0.991] |
| CNN, attention pool | 0.885 [0.873, 0.897] | 0.986 [0.983, 0.988] | pending | pending |
| CNN + ECA, mean pool | 0.896 [0.884, 0.908] | 0.982 [0.980, 0.985] | pending | pending |
| CNN + ECA, attention pool | 0.905 [0.893, 0.917] | 0.989 [0.987, 0.991] | pending | pending |
| CNN + CBAM, mean pool | 0.925 [0.915, 0.934] | 0.989 [0.987, 0.991] | pending | pending |
| CNN + CBAM, attention pool | 0.918 [0.907, 0.928] | 0.988 [0.986, 0.990] | pending | pending |
| small ResNet, mean pool | 0.909 [0.898, 0.919] | 0.981 [0.977, 0.984] | pending | pending |
| small ResNet, attention pool | 0.895 [0.885, 0.905] | 0.984 [0.981, 0.987] | pending | pending |
| small ResNet + ECA, mean pool | 0.907 [0.895, 0.918] | 0.988 [0.985, 0.990] | pending | pending |
| small ResNet + ECA, attention pool | 0.909 [0.899, 0.918] | 0.990 [0.989, 0.992] | pending | pending |
| small ResNet + CBAM, mean pool | 0.912 [0.902, 0.922] | 0.979 [0.975, 0.982] | pending | pending |
| small ResNet + CBAM, attention pool | 0.899 [0.889, 0.909] | 0.984 [0.982, 0.987] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.913 vs 0.885 | +0.028 [+0.014, +0.042] | 3 / 2 |  | 0.987 vs 0.985 | +0.002 [-0.001, +0.006] | 1 / 4 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.885 vs 0.885 | +0.000 [-0.015, +0.015] | 1 / 4 |  | 0.986 vs 0.985 | +0.001 [-0.002, +0.005] | 0 / 5 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.896 vs 0.885 | +0.011 [-0.002, +0.025] | 1 / 4 |  | 0.982 vs 0.985 | -0.002 [-0.005, +0.001] | 0 / 5 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.905 vs 0.885 | +0.020 [+0.004, +0.036] | 0 / 5 |  | 0.989 vs 0.985 | +0.004 [+0.001, +0.008] | 1 / 4 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.925 vs 0.885 | +0.040 [+0.025, +0.055] | 4 / 1 | ✓ | 0.989 vs 0.985 | +0.004 [+0.001, +0.008] | 2 / 3 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.918 vs 0.885 | +0.033 [+0.018, +0.049] | 3 / 2 |  | 0.988 vs 0.985 | +0.003 [-0.000, +0.008] | 0 / 5 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.909 vs 0.885 | +0.024 [+0.010, +0.039] | 2 / 3 |  | 0.981 vs 0.985 | -0.004 [-0.008, +0.000] | 0 / 5 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.895 vs 0.885 | +0.010 [-0.005, +0.025] | 1 / 4 |  | 0.984 vs 0.985 | -0.000 [-0.004, +0.003] | 0 / 5 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.907 vs 0.885 | +0.022 [+0.007, +0.037] | 2 / 3 |  | 0.988 vs 0.985 | +0.003 [-0.000, +0.007] | 0 / 5 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.909 vs 0.885 | +0.024 [+0.009, +0.040] | 5 / 0 | ✓ | 0.990 vs 0.985 | +0.006 [+0.003, +0.010] | 2 / 3 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.912 vs 0.885 | +0.027 [+0.013, +0.042] | 3 / 2 |  | 0.979 vs 0.985 | -0.006 [-0.010, -0.002] | 0 / 5 | ✗ |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.899 vs 0.885 | +0.014 [-0.001, +0.029] | 3 / 2 |  | 0.984 vs 0.985 | -0.000 [-0.004, +0.004] | 1 / 4 |  |
| 1 | CNN, mean pool − logistic regression | RGB | 0.918 vs 0.675 | +0.243 [+0.216, +0.270] | 5 / 0 | ✓ | 0.988 vs 0.864 | +0.125 [+0.110, +0.139] | 5 / 0 | ✓ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.896 vs 0.913 | -0.017 [-0.028, -0.007] | 1 / 4 | ✗ | 0.982 vs 0.987 | -0.005 [-0.006, -0.003] | 1 / 4 | ✗ |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.905 vs 0.885 | +0.020 [+0.008, +0.033] | 1 / 4 |  | 0.989 vs 0.986 | +0.003 [+0.002, +0.005] | 5 / 0 | ✓ |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.925 vs 0.913 | +0.012 [+0.003, +0.022] | 4 / 1 | ✓ | 0.989 vs 0.987 | +0.002 [+0.000, +0.003] | 3 / 2 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.918 vs 0.885 | +0.033 [+0.022, +0.045] | 3 / 2 |  | 0.988 vs 0.986 | +0.003 [+0.000, +0.005] | 5 / 0 | ✓ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.907 vs 0.909 | -0.002 [-0.011, +0.008] | 2 / 3 |  | 0.988 vs 0.981 | +0.007 [+0.004, +0.009] | 4 / 1 | ✓ |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.909 vs 0.895 | +0.014 [+0.005, +0.023] | 4 / 1 | ✓ | 0.990 vs 0.984 | +0.006 [+0.004, +0.008] | 5 / 0 | ✓ |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.912 vs 0.909 | +0.003 [-0.008, +0.013] | 2 / 3 |  | 0.979 vs 0.981 | -0.002 [-0.004, -0.000] | 3 / 2 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.899 vs 0.895 | +0.004 [-0.004, +0.012] | 3 / 2 |  | 0.984 vs 0.984 | +0.000 [-0.002, +0.002] | 4 / 1 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.885 vs 0.913 | -0.028 [-0.040, -0.016] | 1 / 4 | ✗ | 0.986 vs 0.987 | -0.001 [-0.003, +0.000] | 1 / 4 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.905 vs 0.896 | +0.009 [-0.004, +0.022] | 3 / 2 |  | 0.989 vs 0.982 | +0.007 [+0.005, +0.008] | 5 / 0 | ✓ |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.918 vs 0.925 | -0.007 [-0.016, +0.002] | 2 / 3 |  | 0.988 vs 0.989 | -0.001 [-0.003, +0.001] | 2 / 3 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.895 vs 0.909 | -0.014 [-0.024, -0.004] | 3 / 2 |  | 0.984 vs 0.981 | +0.003 [+0.000, +0.006] | 4 / 1 | ✓ |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.909 vs 0.907 | +0.002 [-0.009, +0.012] | 5 / 0 |  | 0.990 vs 0.988 | +0.003 [+0.001, +0.004] | 4 / 1 | ✓ |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.899 vs 0.912 | -0.013 [-0.024, -0.002] | 2 / 3 |  | 0.984 vs 0.979 | +0.006 [+0.003, +0.008] | 4 / 1 | ✓ |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.909 vs 0.913 | -0.004 [-0.013, +0.004] | 2 / 3 |  | 0.981 vs 0.987 | -0.006 [-0.008, -0.004] | 0 / 5 | ✗ |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.895 vs 0.885 | +0.010 [+0.000, +0.020] | 4 / 1 | ✓ | 0.984 vs 0.986 | -0.001 [-0.003, +0.001] | 4 / 1 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.907 vs 0.896 | +0.011 [+0.000, +0.022] | 5 / 0 | ✓ | 0.988 vs 0.982 | +0.005 [+0.004, +0.007] | 4 / 1 | ✓ |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.909 vs 0.905 | +0.004 [-0.008, +0.015] | 5 / 0 |  | 0.990 vs 0.989 | +0.001 [-0.000, +0.003] | 4 / 1 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.912 vs 0.925 | -0.013 [-0.023, -0.003] | 2 / 3 |  | 0.979 vs 0.989 | -0.010 [-0.013, -0.008] | 0 / 5 | ✗ |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.899 vs 0.918 | -0.019 [-0.030, -0.009] | 1 / 4 | ✗ | 0.984 vs 0.988 | -0.004 [-0.006, -0.002] | 2 / 3 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.885 vs 0.675 | +0.210 [+0.183, +0.237] | single fits | ✓ | 0.985 vs 0.864 | +0.121 [+0.107, +0.135] | single fits | ✓ |
| 4 | CNN, mean pool − CNN, mean pool | 204 bands − RGB | 0.913 vs 0.918 | -0.005 [-0.019, +0.010] | 0 / 5 |  | 0.987 vs 0.988 | -0.001 [-0.004, +0.002] | 1 / 4 |  |

Marked, of 35 comparisons: balanced accuracy 8 ✓ and 3 ✗; AUC 11 ✓ and 4 ✗. Chance alone would put about 1.8 intervals per score off 0 (fewer once the seeds must agree).

## Disease

Leafroll leaves out Cab Franc everywhere (1 of its 250 leaves has leafroll). Red blotch on all 998 is also scored without Cab Franc's leaves (241 of its 250 leaves have red blotch), with the same models. The **variety-only rule** predicts from the leaf's variety alone (each variety's disease rate on the training rows); a pooled model has to beat it to show more than the varieties' different disease rates. Its interval has zero width: it gives every leaf of one variety in one test row the same P, and the resamples keep the number of positive and negative leaves in each variety × row, so every resample gives it the same score.

### Leafroll overview

Balanced accuracy · AUC of every model in every setting; intervals and comparisons in the sections below.

204 bands:

| model | Baco | Chardonnay | Vidal | pooled, Baco + Chardonnay + Vidal |
|---|---|---|---|---|
| logistic regression | 0.438 · 0.468 | 0.533 · 0.528 | 0.457 · 0.432 | 0.587 · 0.619 |
| CNN, mean pool | 0.497 · 0.521 | 0.569 · 0.563 | 0.454 · 0.424 | 0.598 · 0.635 |
| CNN, attention pool | 0.476 · 0.529 | 0.553 · 0.545 | 0.461 · 0.435 | 0.595 · 0.624 |
| CNN + ECA, mean pool | 0.590 · 0.553 | 0.554 · 0.581 | 0.400 · 0.414 | 0.606 · 0.636 |
| CNN + ECA, attention pool | 0.572 · 0.559 | 0.580 · 0.558 | 0.457 · 0.417 | 0.607 · 0.633 |
| CNN + CBAM, mean pool | 0.594 · 0.574 | 0.535 · 0.555 | 0.478 · 0.430 | 0.612 · 0.651 |
| CNN + CBAM, attention pool | 0.555 · 0.599 | 0.561 · 0.571 | 0.411 · 0.406 | 0.626 · 0.666 |
| small ResNet, mean pool | 0.550 · 0.561 | 0.564 · 0.568 | 0.492 · 0.469 | 0.584 · 0.632 |
| small ResNet, attention pool | 0.585 · 0.568 | 0.561 · 0.552 | 0.482 · 0.461 | 0.611 · 0.661 |
| small ResNet + ECA, mean pool | 0.559 · 0.550 | 0.573 · 0.562 | 0.453 · 0.428 | 0.604 · 0.654 |
| small ResNet + ECA, attention pool | 0.565 · 0.564 | 0.545 · 0.572 | 0.453 · 0.448 | 0.574 · 0.622 |
| small ResNet + CBAM, mean pool | 0.553 · 0.574 | 0.567 · 0.570 | 0.417 · 0.457 | 0.612 · 0.637 |
| small ResNet + CBAM, attention pool | 0.545 · 0.544 | 0.560 · 0.547 | 0.437 · 0.447 | 0.585 · 0.626 |
| variety-only rule |  |  |  | 0.676 · 0.652 |

RGB:

| model | Baco | Chardonnay | Vidal | pooled, Baco + Chardonnay + Vidal |
|---|---|---|---|---|
| logistic regression | 0.427 · 0.410 | 0.491 · 0.470 | 0.422 · 0.411 | 0.573 · 0.622 |
| CNN, mean pool | pending | pending | pending | pending |
| CNN, attention pool | pending | pending | pending | pending |
| CNN + ECA, mean pool | pending | pending | pending | pending |
| CNN + ECA, attention pool | pending | pending | pending | pending |
| CNN + CBAM, mean pool | pending | pending | pending | pending |
| CNN + CBAM, attention pool | pending | pending | pending | pending |
| small ResNet, mean pool | pending | pending | pending | pending |
| small ResNet, attention pool | pending | pending | pending | pending |
| small ResNet + ECA, mean pool | pending | pending | pending | pending |
| small ResNet + ECA, attention pool | pending | pending | pending | pending |
| small ResNet + CBAM, mean pool | pending | pending | pending | pending |
| small ResNet + CBAM, attention pool | pending | pending | pending | pending |
| variety-only rule |  |  |  | 0.676 · 0.652 |

### Red blotch overview

Balanced accuracy · AUC of every model in every setting; intervals and comparisons in the sections below.

204 bands:

| model | Baco | Chardonnay | Vidal | pooled, all 998 leaves | pooled, trained without Cab Franc | pooled, all 998 leaves, scored without Cab Franc |
|---|---|---|---|---|---|---|
| logistic regression | 0.548 · 0.526 | 0.597 · 0.566 | 0.538 · 0.566 | 0.663 · 0.724 | 0.580 · 0.638 | 0.590 · 0.634 |
| CNN, mean pool | 0.500 · 0.497 | 0.523 · 0.540 | 0.518 · 0.542 | 0.693 · 0.748 | 0.604 · 0.642 | 0.607 · 0.649 |
| CNN, attention pool | 0.485 · 0.493 | 0.512 · 0.536 | 0.493 · 0.538 | 0.677 · 0.748 | 0.608 · 0.649 | 0.586 · 0.652 |
| CNN + ECA, mean pool | 0.483 · 0.519 | 0.546 · 0.586 | 0.486 · 0.559 | 0.685 · 0.752 | 0.596 · 0.622 | 0.596 · 0.647 |
| CNN + ECA, attention pool | 0.521 · 0.524 | 0.572 · 0.539 | 0.510 · 0.535 | 0.683 · 0.741 | 0.625 · 0.647 | 0.598 · 0.639 |
| CNN + CBAM, mean pool | 0.471 · 0.471 | 0.560 · 0.577 | 0.549 · 0.574 | 0.702 · 0.763 | 0.607 · 0.635 | 0.616 · 0.664 |
| CNN + CBAM, attention pool | 0.518 · 0.480 | 0.591 · 0.584 | 0.480 · 0.519 | 0.699 · 0.763 | 0.590 · 0.633 | 0.601 · 0.657 |
| small ResNet, mean pool | 0.508 · 0.483 | 0.589 · 0.571 | 0.546 · 0.577 | 0.691 · 0.755 | 0.604 · 0.651 | 0.601 · 0.660 |
| small ResNet, attention pool | 0.496 · 0.485 | 0.542 · 0.543 | 0.495 · 0.529 | 0.686 · 0.752 | 0.587 · 0.641 | 0.591 · 0.651 |
| small ResNet + ECA, mean pool | 0.509 · 0.522 | 0.590 · 0.559 | 0.561 · 0.587 | 0.691 · 0.755 | 0.616 · 0.652 | 0.597 · 0.653 |
| small ResNet + ECA, attention pool | 0.489 · 0.498 | 0.595 · 0.564 | 0.500 · 0.524 | 0.668 · 0.747 | 0.615 · 0.652 | 0.581 · 0.645 |
| small ResNet + CBAM, mean pool | 0.484 · 0.495 | 0.572 · 0.559 | 0.528 · 0.543 | 0.690 · 0.754 | 0.609 · 0.648 | 0.592 · 0.649 |
| small ResNet + CBAM, attention pool | 0.504 · 0.492 | 0.554 · 0.541 | 0.510 · 0.489 | 0.689 · 0.760 | 0.617 · 0.651 | 0.612 · 0.663 |
| variety-only rule |  |  |  | 0.706 · 0.764 | 0.619 · 0.650 | 0.609 · 0.645 |

RGB:

| model | Baco | Chardonnay | Vidal | pooled, all 998 leaves | pooled, trained without Cab Franc | pooled, all 998 leaves, scored without Cab Franc |
|---|---|---|---|---|---|---|
| logistic regression | 0.559 · 0.522 | 0.634 · 0.656 | 0.528 · 0.607 | 0.641 · 0.683 | 0.610 · 0.651 | 0.585 · 0.615 |
| CNN, mean pool | pending | pending | pending | pending | pending | pending |
| CNN, attention pool | pending | pending | pending | pending | pending | pending |
| CNN + ECA, mean pool | pending | pending | pending | pending | pending | pending |
| CNN + ECA, attention pool | pending | pending | pending | pending | pending | pending |
| CNN + CBAM, mean pool | pending | pending | pending | pending | pending | pending |
| CNN + CBAM, attention pool | pending | pending | pending | pending | pending | pending |
| small ResNet, mean pool | pending | pending | pending | pending | pending | pending |
| small ResNet, attention pool | pending | pending | pending | pending | pending | pending |
| small ResNet + ECA, mean pool | pending | pending | pending | pending | pending | pending |
| small ResNet + ECA, attention pool | pending | pending | pending | pending | pending | pending |
| small ResNet + CBAM, mean pool | pending | pending | pending | pending | pending | pending |
| small ResNet + CBAM, attention pool | pending | pending | pending | pending | pending | pending |
| variety-only rule |  |  |  | 0.706 · 0.764 | 0.619 · 0.650 | 0.609 · 0.645 |

### Leafroll, Baco

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.438 [0.385, 0.493] | 0.468 [0.402, 0.538] | 0.427 [0.380, 0.474] | 0.410 [0.373, 0.451] |
| CNN, mean pool | 0.497 [0.468, 0.528] | 0.521 [0.500, 0.544] | pending | pending |
| CNN, attention pool | 0.476 [0.446, 0.505] | 0.529 [0.508, 0.552] | pending | pending |
| CNN + ECA, mean pool | 0.590 [0.573, 0.607] | 0.553 [0.533, 0.573] | pending | pending |
| CNN + ECA, attention pool | 0.572 [0.546, 0.600] | 0.559 [0.539, 0.579] | pending | pending |
| CNN + CBAM, mean pool | 0.594 [0.570, 0.619] | 0.574 [0.552, 0.597] | pending | pending |
| CNN + CBAM, attention pool | 0.555 [0.516, 0.596] | 0.599 [0.567, 0.632] | pending | pending |
| small ResNet, mean pool | 0.550 [0.526, 0.575] | 0.561 [0.543, 0.580] | pending | pending |
| small ResNet, attention pool | 0.585 [0.552, 0.615] | 0.568 [0.548, 0.589] | pending | pending |
| small ResNet + ECA, mean pool | 0.559 [0.517, 0.601] | 0.550 [0.520, 0.581] | pending | pending |
| small ResNet + ECA, attention pool | 0.565 [0.543, 0.586] | 0.564 [0.547, 0.580] | pending | pending |
| small ResNet + CBAM, mean pool | 0.553 [0.546, 0.556] | 0.574 [0.556, 0.594] | pending | pending |
| small ResNet + CBAM, attention pool | 0.545 [0.505, 0.584] | 0.544 [0.516, 0.573] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.497 vs 0.438 | +0.060 [-0.002, +0.119] | 5 / 0 |  | 0.521 vs 0.468 | +0.054 [-0.011, +0.116] | 4 / 1 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.476 vs 0.438 | +0.038 [-0.020, +0.093] | 5 / 0 |  | 0.529 vs 0.468 | +0.061 [-0.002, +0.123] | 5 / 0 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.590 vs 0.438 | +0.153 [+0.096, +0.209] | 5 / 0 | ✓ | 0.553 vs 0.468 | +0.085 [+0.016, +0.149] | 5 / 0 | ✓ |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.572 vs 0.438 | +0.135 [+0.069, +0.200] | 5 / 0 | ✓ | 0.559 vs 0.468 | +0.091 [+0.023, +0.155] | 5 / 0 | ✓ |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.594 vs 0.438 | +0.157 [+0.087, +0.226] | 5 / 0 | ✓ | 0.574 vs 0.468 | +0.106 [+0.039, +0.170] | 5 / 0 | ✓ |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.555 vs 0.438 | +0.117 [+0.044, +0.188] | 5 / 0 | ✓ | 0.599 vs 0.468 | +0.131 [+0.061, +0.200] | 5 / 0 | ✓ |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.550 vs 0.438 | +0.113 [+0.049, +0.174] | 5 / 0 | ✓ | 0.561 vs 0.468 | +0.093 [+0.028, +0.154] | 4 / 1 | ✓ |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.585 vs 0.438 | +0.148 [+0.086, +0.208] | 5 / 0 | ✓ | 0.568 vs 0.468 | +0.100 [+0.038, +0.159] | 5 / 0 | ✓ |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.559 vs 0.438 | +0.122 [+0.049, +0.193] | 5 / 0 | ✓ | 0.550 vs 0.468 | +0.082 [+0.011, +0.149] | 5 / 0 | ✓ |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.565 vs 0.438 | +0.127 [+0.068, +0.182] | 4 / 1 | ✓ | 0.564 vs 0.468 | +0.096 [+0.031, +0.156] | 4 / 1 | ✓ |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.553 vs 0.438 | +0.115 [+0.060, +0.168] | 5 / 0 | ✓ | 0.574 vs 0.468 | +0.106 [+0.045, +0.165] | 5 / 0 | ✓ |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.545 vs 0.438 | +0.107 [+0.044, +0.173] | 5 / 0 | ✓ | 0.544 vs 0.468 | +0.076 [+0.011, +0.138] | 5 / 0 | ✓ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.590 vs 0.497 | +0.093 [+0.060, +0.126] | 3 / 2 |  | 0.553 vs 0.521 | +0.031 [+0.008, +0.054] | 4 / 1 | ✓ |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.572 vs 0.476 | +0.097 [+0.064, +0.133] | 4 / 1 | ✓ | 0.559 vs 0.529 | +0.030 [+0.009, +0.050] | 4 / 1 | ✓ |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.594 vs 0.497 | +0.097 [+0.059, +0.133] | 4 / 1 | ✓ | 0.574 vs 0.521 | +0.052 [+0.026, +0.079] | 4 / 1 | ✓ |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.555 vs 0.476 | +0.079 [+0.030, +0.130] | 3 / 2 |  | 0.599 vs 0.529 | +0.070 [+0.039, +0.101] | 4 / 1 | ✓ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.559 vs 0.550 | +0.009 [-0.035, +0.051] | 3 / 2 |  | 0.550 vs 0.561 | -0.011 [-0.037, +0.015] | 2 / 3 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.565 vs 0.585 | -0.021 [-0.052, +0.012] | 2 / 3 |  | 0.564 vs 0.568 | -0.004 [-0.023, +0.015] | 2 / 3 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.553 vs 0.550 | +0.002 [-0.021, +0.026] | 3 / 2 |  | 0.574 vs 0.561 | +0.013 [-0.004, +0.031] | 5 / 0 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.545 vs 0.585 | -0.040 [-0.086, +0.003] | 3 / 2 |  | 0.544 vs 0.568 | -0.024 [-0.049, +0.002] | 2 / 3 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.476 vs 0.497 | -0.022 [-0.063, +0.022] | 4 / 1 |  | 0.529 vs 0.521 | +0.008 [-0.015, +0.030] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.572 vs 0.590 | -0.018 [-0.049, +0.015] | 2 / 3 |  | 0.559 vs 0.553 | +0.006 [-0.014, +0.027] | 2 / 3 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.555 vs 0.594 | -0.039 [-0.077, -0.000] | 1 / 4 | ✗ | 0.599 vs 0.574 | +0.025 [-0.001, +0.052] | 3 / 2 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.585 vs 0.550 | +0.035 [-0.003, +0.071] | 4 / 1 |  | 0.568 vs 0.561 | +0.007 [-0.014, +0.028] | 2 / 3 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.565 vs 0.559 | +0.005 [-0.039, +0.049] | 3 / 2 |  | 0.564 vs 0.550 | +0.014 [-0.018, +0.044] | 3 / 2 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.545 vs 0.553 | -0.008 [-0.048, +0.030] | 3 / 2 |  | 0.544 vs 0.574 | -0.031 [-0.057, -0.002] | 1 / 4 | ✗ |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.550 vs 0.497 | +0.053 [+0.015, +0.089] | 4 / 1 | ✓ | 0.561 vs 0.521 | +0.039 [+0.019, +0.061] | 4 / 1 | ✓ |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.585 vs 0.476 | +0.110 [+0.068, +0.151] | 3 / 2 |  | 0.568 vs 0.529 | +0.038 [+0.018, +0.058] | 4 / 1 | ✓ |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.559 vs 0.590 | -0.031 [-0.075, +0.010] | 2 / 3 |  | 0.550 vs 0.553 | -0.003 [-0.034, +0.029] | 1 / 4 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.565 vs 0.572 | -0.008 [-0.042, +0.023] | 3 / 2 |  | 0.564 vs 0.559 | +0.005 [-0.019, +0.030] | 3 / 2 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.553 vs 0.594 | -0.041 [-0.066, -0.016] | 1 / 4 | ✗ | 0.574 vs 0.574 | +0.001 [-0.024, +0.024] | 3 / 2 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.545 vs 0.555 | -0.010 [-0.058, +0.034] | 3 / 2 |  | 0.544 vs 0.599 | -0.055 [-0.086, -0.024] | 1 / 4 | ✗ |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.438 vs 0.427 | +0.010 [-0.045, +0.064] | single fits |  | 0.468 vs 0.410 | +0.057 [-0.002, +0.121] | single fits |  |

Marked, of 33 comparisons: balanced accuracy 13 ✓ and 2 ✗; AUC 16 ✓ and 2 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Leafroll, Chardonnay

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.533 [0.473, 0.593] | 0.528 [0.458, 0.599] | 0.491 [0.431, 0.551] | 0.470 [0.400, 0.541] |
| CNN, mean pool | 0.569 [0.537, 0.601] | 0.563 [0.532, 0.594] | pending | pending |
| CNN, attention pool | 0.553 [0.521, 0.585] | 0.545 [0.521, 0.570] | pending | pending |
| CNN + ECA, mean pool | 0.554 [0.526, 0.582] | 0.581 [0.551, 0.609] | pending | pending |
| CNN + ECA, attention pool | 0.580 [0.548, 0.609] | 0.558 [0.527, 0.587] | pending | pending |
| CNN + CBAM, mean pool | 0.535 [0.499, 0.568] | 0.555 [0.522, 0.585] | pending | pending |
| CNN + CBAM, attention pool | 0.561 [0.533, 0.593] | 0.571 [0.539, 0.601] | pending | pending |
| small ResNet, mean pool | 0.564 [0.536, 0.592] | 0.568 [0.538, 0.597] | pending | pending |
| small ResNet, attention pool | 0.561 [0.533, 0.589] | 0.552 [0.531, 0.573] | pending | pending |
| small ResNet + ECA, mean pool | 0.573 [0.542, 0.601] | 0.562 [0.538, 0.588] | pending | pending |
| small ResNet + ECA, attention pool | 0.545 [0.516, 0.576] | 0.572 [0.541, 0.601] | pending | pending |
| small ResNet + CBAM, mean pool | 0.567 [0.538, 0.595] | 0.570 [0.541, 0.600] | pending | pending |
| small ResNet + CBAM, attention pool | 0.560 [0.528, 0.591] | 0.547 [0.521, 0.572] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.569 vs 0.533 | +0.036 [-0.028, +0.100] | 4 / 1 |  | 0.563 vs 0.528 | +0.035 [-0.029, +0.099] | 3 / 2 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.553 vs 0.533 | +0.020 [-0.051, +0.088] | 2 / 3 |  | 0.545 vs 0.528 | +0.017 [-0.053, +0.088] | 4 / 1 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.554 vs 0.533 | +0.021 [-0.043, +0.081] | 3 / 2 |  | 0.581 vs 0.528 | +0.053 [-0.009, +0.116] | 5 / 0 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.580 vs 0.533 | +0.047 [-0.012, +0.103] | 5 / 0 |  | 0.558 vs 0.528 | +0.030 [-0.034, +0.093] | 4 / 1 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.535 vs 0.533 | +0.002 [-0.064, +0.070] | 4 / 1 |  | 0.555 vs 0.528 | +0.027 [-0.039, +0.090] | 3 / 2 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.561 vs 0.533 | +0.028 [-0.036, +0.092] | 5 / 0 |  | 0.571 vs 0.528 | +0.043 [-0.024, +0.109] | 5 / 0 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.564 vs 0.533 | +0.031 [-0.033, +0.092] | 4 / 1 |  | 0.568 vs 0.528 | +0.040 [-0.025, +0.104] | 5 / 0 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.561 vs 0.533 | +0.028 [-0.036, +0.092] | 4 / 1 |  | 0.552 vs 0.528 | +0.024 [-0.047, +0.094] | 5 / 0 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.573 vs 0.533 | +0.040 [-0.021, +0.100] | 3 / 2 |  | 0.562 vs 0.528 | +0.034 [-0.032, +0.102] | 4 / 1 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.545 vs 0.533 | +0.012 [-0.054, +0.080] | 3 / 2 |  | 0.572 vs 0.528 | +0.043 [-0.024, +0.112] | 4 / 1 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.567 vs 0.533 | +0.034 [-0.030, +0.094] | 5 / 0 |  | 0.570 vs 0.528 | +0.042 [-0.025, +0.108] | 4 / 1 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.560 vs 0.533 | +0.027 [-0.033, +0.086] | 5 / 0 |  | 0.547 vs 0.528 | +0.019 [-0.048, +0.085] | 4 / 1 |  |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.554 vs 0.569 | -0.015 [-0.050, +0.021] | 2 / 3 |  | 0.581 vs 0.563 | +0.018 [-0.009, +0.043] | 4 / 1 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.580 vs 0.553 | +0.027 [-0.021, +0.075] | 4 / 1 |  | 0.558 vs 0.545 | +0.013 [-0.012, +0.037] | 3 / 2 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.535 vs 0.569 | -0.034 [-0.078, +0.010] | 3 / 2 |  | 0.555 vs 0.563 | -0.008 [-0.037, +0.022] | 2 / 3 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.561 vs 0.553 | +0.008 [-0.028, +0.048] | 3 / 2 |  | 0.571 vs 0.545 | +0.026 [-0.004, +0.055] | 4 / 1 |  |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.573 vs 0.564 | +0.009 [-0.020, +0.037] | 2 / 3 |  | 0.562 vs 0.568 | -0.005 [-0.026, +0.017] | 2 / 3 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.545 vs 0.561 | -0.016 [-0.047, +0.012] | 1 / 4 |  | 0.572 vs 0.552 | +0.019 [-0.005, +0.043] | 4 / 1 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.567 vs 0.564 | +0.002 [-0.030, +0.034] | 2 / 3 |  | 0.570 vs 0.568 | +0.003 [-0.020, +0.025] | 3 / 2 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.560 vs 0.561 | -0.001 [-0.042, +0.038] | 3 / 2 |  | 0.547 vs 0.552 | -0.005 [-0.028, +0.017] | 4 / 1 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.553 vs 0.569 | -0.016 [-0.056, +0.025] | 1 / 4 |  | 0.545 vs 0.563 | -0.018 [-0.040, +0.005] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.580 vs 0.554 | +0.026 [-0.002, +0.058] | 4 / 1 |  | 0.558 vs 0.581 | -0.023 [-0.045, -0.000] | 1 / 4 | ✗ |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.561 vs 0.535 | +0.026 [-0.014, +0.066] | 2 / 3 |  | 0.571 vs 0.555 | +0.016 [-0.007, +0.038] | 2 / 3 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.561 vs 0.564 | -0.003 [-0.027, +0.020] | 3 / 2 |  | 0.552 vs 0.568 | -0.015 [-0.035, +0.005] | 0 / 5 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.545 vs 0.573 | -0.028 [-0.064, +0.008] | 3 / 2 |  | 0.572 vs 0.562 | +0.009 [-0.013, +0.030] | 3 / 2 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.560 vs 0.567 | -0.007 [-0.031, +0.017] | 0 / 5 |  | 0.547 vs 0.570 | -0.023 [-0.043, -0.004] | 2 / 3 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.564 vs 0.569 | -0.004 [-0.036, +0.024] | 2 / 3 |  | 0.568 vs 0.563 | +0.005 [-0.016, +0.025] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.561 vs 0.553 | +0.008 [-0.023, +0.039] | 2 / 3 |  | 0.552 vs 0.545 | +0.007 [-0.010, +0.025] | 1 / 4 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.573 vs 0.554 | +0.019 [-0.013, +0.054] | 3 / 2 |  | 0.562 vs 0.581 | -0.019 [-0.043, +0.007] | 2 / 3 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.545 vs 0.580 | -0.036 [-0.080, +0.012] | 2 / 3 |  | 0.572 vs 0.558 | +0.013 [-0.011, +0.038] | 3 / 2 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.567 vs 0.535 | +0.032 [-0.005, +0.072] | 4 / 1 |  | 0.570 vs 0.555 | +0.015 [-0.015, +0.045] | 3 / 2 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.560 vs 0.561 | -0.002 [-0.038, +0.034] | 2 / 3 |  | 0.547 vs 0.571 | -0.024 [-0.049, +0.000] | 2 / 3 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.533 vs 0.491 | +0.042 [-0.021, +0.107] | single fits |  | 0.528 vs 0.470 | +0.058 [-0.017, +0.131] | single fits |  |

Marked, of 33 comparisons: balanced accuracy 0 ✓ and 0 ✗; AUC 0 ✓ and 1 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Leafroll, Vidal

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.457 [0.370, 0.543] | 0.432 [0.335, 0.543] | 0.422 [0.336, 0.510] | 0.411 [0.300, 0.529] |
| CNN, mean pool | 0.454 [0.394, 0.522] | 0.424 [0.374, 0.473] | pending | pending |
| CNN, attention pool | 0.461 [0.385, 0.542] | 0.435 [0.374, 0.500] | pending | pending |
| CNN + ECA, mean pool | 0.400 [0.355, 0.452] | 0.414 [0.363, 0.474] | pending | pending |
| CNN + ECA, attention pool | 0.457 [0.386, 0.536] | 0.417 [0.356, 0.482] | pending | pending |
| CNN + CBAM, mean pool | 0.478 [0.420, 0.543] | 0.430 [0.365, 0.496] | pending | pending |
| CNN + CBAM, attention pool | 0.411 [0.342, 0.485] | 0.406 [0.343, 0.474] | pending | pending |
| small ResNet, mean pool | 0.492 [0.419, 0.569] | 0.469 [0.399, 0.547] | pending | pending |
| small ResNet, attention pool | 0.482 [0.412, 0.552] | 0.461 [0.394, 0.526] | pending | pending |
| small ResNet + ECA, mean pool | 0.453 [0.381, 0.518] | 0.428 [0.357, 0.500] | pending | pending |
| small ResNet + ECA, attention pool | 0.453 [0.384, 0.522] | 0.448 [0.377, 0.524] | pending | pending |
| small ResNet + CBAM, mean pool | 0.417 [0.349, 0.484] | 0.457 [0.382, 0.533] | pending | pending |
| small ResNet + CBAM, attention pool | 0.437 [0.379, 0.502] | 0.447 [0.366, 0.531] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.454 vs 0.457 | -0.003 [-0.094, +0.095] | 1 / 4 |  | 0.424 vs 0.432 | -0.008 [-0.116, +0.090] | 4 / 1 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.461 vs 0.457 | +0.004 [-0.103, +0.111] | 2 / 3 |  | 0.435 vs 0.432 | +0.003 [-0.116, +0.116] | 3 / 2 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.400 vs 0.457 | -0.057 [-0.170, +0.058] | 2 / 3 |  | 0.414 vs 0.432 | -0.017 [-0.142, +0.105] | 2 / 2 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.457 vs 0.457 | +0.000 [-0.121, +0.122] | 3 / 2 |  | 0.417 vs 0.432 | -0.015 [-0.131, +0.097] | 3 / 2 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.478 vs 0.457 | +0.020 [-0.104, +0.154] | 2 / 3 |  | 0.430 vs 0.432 | -0.002 [-0.128, +0.129] | 4 / 1 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.411 vs 0.457 | -0.047 [-0.164, +0.067] | 4 / 1 |  | 0.406 vs 0.432 | -0.025 [-0.139, +0.083] | 4 / 1 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.492 vs 0.457 | +0.035 [-0.056, +0.119] | 3 / 2 |  | 0.469 vs 0.432 | +0.038 [-0.075, +0.144] | 3 / 2 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.482 vs 0.457 | +0.025 [-0.068, +0.118] | 3 / 2 |  | 0.461 vs 0.432 | +0.029 [-0.080, +0.125] | 4 / 1 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.453 vs 0.457 | -0.004 [-0.120, +0.104] | 3 / 2 |  | 0.428 vs 0.432 | -0.004 [-0.133, +0.114] | 4 / 1 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.453 vs 0.457 | -0.004 [-0.119, +0.104] | 3 / 2 |  | 0.448 vs 0.432 | +0.016 [-0.111, +0.135] | 4 / 1 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.417 vs 0.457 | -0.040 [-0.132, +0.050] | 3 / 2 |  | 0.457 vs 0.432 | +0.025 [-0.087, +0.130] | 4 / 1 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.437 vs 0.457 | -0.020 [-0.110, +0.061] | 1 / 4 |  | 0.447 vs 0.432 | +0.015 [-0.107, +0.126] | 5 / 0 |  |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.400 vs 0.454 | -0.054 [-0.144, +0.043] | 3 / 2 |  | 0.414 vs 0.424 | -0.009 [-0.072, +0.061] | 2 / 3 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.457 vs 0.461 | -0.004 [-0.075, +0.071] | 3 / 2 |  | 0.417 vs 0.435 | -0.019 [-0.059, +0.027] | 1 / 4 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.478 vs 0.454 | +0.023 [-0.075, +0.126] | 3 / 2 |  | 0.430 vs 0.424 | +0.006 [-0.069, +0.086] | 2 / 3 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.411 vs 0.461 | -0.050 [-0.132, +0.038] | 5 / 0 |  | 0.406 vs 0.435 | -0.029 [-0.094, +0.042] | 2 / 3 |  |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.453 vs 0.492 | -0.038 [-0.124, +0.051] | 2 / 3 |  | 0.428 vs 0.469 | -0.041 [-0.113, +0.024] | 2 / 3 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.453 vs 0.482 | -0.029 [-0.108, +0.053] | 3 / 2 |  | 0.448 vs 0.461 | -0.013 [-0.060, +0.040] | 3 / 2 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.417 vs 0.492 | -0.074 [-0.135, -0.027] | 1 / 4 | ✗ | 0.457 vs 0.469 | -0.012 [-0.073, +0.036] | 3 / 2 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.437 vs 0.482 | -0.045 [-0.103, +0.024] | 2 / 3 |  | 0.447 vs 0.461 | -0.014 [-0.070, +0.038] | 4 / 1 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.461 vs 0.454 | +0.007 [-0.081, +0.100] | 2 / 3 |  | 0.435 vs 0.424 | +0.011 [-0.040, +0.062] | 3 / 2 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.457 vs 0.400 | +0.057 [-0.003, +0.119] | 2 / 3 |  | 0.417 vs 0.414 | +0.002 [-0.049, +0.055] | 4 / 1 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.411 vs 0.478 | -0.067 [-0.140, +0.010] | 5 / 0 |  | 0.406 vs 0.430 | -0.023 [-0.092, +0.046] | 3 / 2 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.482 vs 0.492 | -0.009 [-0.074, +0.043] | 2 / 3 |  | 0.461 vs 0.469 | -0.009 [-0.060, +0.037] | 2 / 3 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.453 vs 0.453 | -0.000 [-0.061, +0.056] | 3 / 2 |  | 0.448 vs 0.428 | +0.020 [-0.028, +0.073] | 2 / 3 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.437 vs 0.417 | +0.020 [-0.033, +0.086] | 1 / 4 |  | 0.447 vs 0.457 | -0.010 [-0.061, +0.046] | 4 / 1 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.492 vs 0.454 | +0.037 [-0.037, +0.122] | 3 / 2 |  | 0.469 vs 0.424 | +0.046 [-0.007, +0.106] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.482 vs 0.461 | +0.021 [-0.045, +0.088] | 3 / 2 |  | 0.461 vs 0.435 | +0.025 [-0.033, +0.087] | 1 / 4 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.453 vs 0.400 | +0.053 [-0.012, +0.120] | 3 / 2 |  | 0.428 vs 0.414 | +0.013 [-0.045, +0.073] | 4 / 1 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.453 vs 0.457 | -0.004 [-0.054, +0.050] | 3 / 2 |  | 0.448 vs 0.417 | +0.031 [-0.009, +0.073] | 4 / 1 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.417 vs 0.478 | -0.060 [-0.166, +0.039] | 2 / 3 |  | 0.457 vs 0.430 | +0.027 [-0.075, +0.122] | 3 / 2 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.437 vs 0.411 | +0.027 [-0.074, +0.129] | 0 / 5 |  | 0.447 vs 0.406 | +0.040 [-0.051, +0.130] | 4 / 1 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.457 vs 0.422 | +0.035 [-0.060, +0.126] | single fits |  | 0.432 vs 0.411 | +0.021 [-0.114, +0.152] | single fits |  |

Marked, of 33 comparisons: balanced accuracy 0 ✓ and 1 ✗; AUC 0 ✓ and 0 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Leafroll, pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc)

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.587 [0.557, 0.616] | 0.619 [0.594, 0.644] | 0.573 [0.546, 0.599] | 0.622 [0.595, 0.647] |
| variety-only rule | 0.676 [0.676, 0.676] | 0.652 [0.652, 0.652] | 0.676 [0.676, 0.676] | 0.652 [0.652, 0.652] |
| CNN, mean pool | 0.598 [0.579, 0.617] | 0.635 [0.617, 0.654] | pending | pending |
| CNN, attention pool | 0.595 [0.576, 0.616] | 0.624 [0.603, 0.644] | pending | pending |
| CNN + ECA, mean pool | 0.606 [0.586, 0.627] | 0.636 [0.616, 0.656] | pending | pending |
| CNN + ECA, attention pool | 0.607 [0.589, 0.625] | 0.633 [0.615, 0.651] | pending | pending |
| CNN + CBAM, mean pool | 0.612 [0.591, 0.631] | 0.651 [0.630, 0.671] | pending | pending |
| CNN + CBAM, attention pool | 0.626 [0.604, 0.647] | 0.666 [0.648, 0.684] | pending | pending |
| small ResNet, mean pool | 0.584 [0.563, 0.604] | 0.632 [0.612, 0.652] | pending | pending |
| small ResNet, attention pool | 0.611 [0.590, 0.635] | 0.661 [0.640, 0.682] | pending | pending |
| small ResNet + ECA, mean pool | 0.604 [0.584, 0.624] | 0.654 [0.636, 0.674] | pending | pending |
| small ResNet + ECA, attention pool | 0.574 [0.554, 0.593] | 0.622 [0.603, 0.641] | pending | pending |
| small ResNet + CBAM, mean pool | 0.612 [0.595, 0.629] | 0.637 [0.621, 0.654] | pending | pending |
| small ResNet + CBAM, attention pool | 0.585 [0.567, 0.602] | 0.626 [0.608, 0.645] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **1b** against the variety-only rule (pooled sets); **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.598 vs 0.587 | +0.011 [-0.020, +0.042] | 2 / 3 |  | 0.635 vs 0.619 | +0.017 [-0.006, +0.039] | 3 / 2 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.595 vs 0.587 | +0.008 [-0.022, +0.040] | 1 / 4 |  | 0.624 vs 0.619 | +0.005 [-0.017, +0.026] | 0 / 5 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.606 vs 0.587 | +0.019 [-0.011, +0.050] | 1 / 4 |  | 0.636 vs 0.619 | +0.017 [-0.005, +0.041] | 2 / 3 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.607 vs 0.587 | +0.020 [-0.011, +0.052] | 1 / 4 |  | 0.633 vs 0.619 | +0.014 [-0.007, +0.036] | 2 / 3 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.612 vs 0.587 | +0.025 [-0.006, +0.056] | 3 / 2 |  | 0.651 vs 0.619 | +0.032 [+0.008, +0.055] | 2 / 3 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.626 vs 0.587 | +0.039 [+0.008, +0.070] | 4 / 1 | ✓ | 0.666 vs 0.619 | +0.048 [+0.024, +0.070] | 5 / 0 | ✓ |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.584 vs 0.587 | -0.003 [-0.035, +0.031] | 2 / 3 |  | 0.632 vs 0.619 | +0.013 [-0.011, +0.037] | 1 / 4 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.611 vs 0.587 | +0.024 [-0.007, +0.057] | 4 / 1 |  | 0.661 vs 0.619 | +0.042 [+0.019, +0.064] | 5 / 0 | ✓ |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.604 vs 0.587 | +0.017 [-0.014, +0.050] | 3 / 2 |  | 0.654 vs 0.619 | +0.036 [+0.012, +0.061] | 3 / 2 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.574 vs 0.587 | -0.013 [-0.044, +0.019] | 0 / 5 |  | 0.622 vs 0.619 | +0.004 [-0.020, +0.026] | 2 / 3 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.612 vs 0.587 | +0.025 [-0.004, +0.057] | 3 / 2 |  | 0.637 vs 0.619 | +0.018 [-0.004, +0.040] | 3 / 2 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.585 vs 0.587 | -0.003 [-0.034, +0.029] | 0 / 5 |  | 0.626 vs 0.619 | +0.007 [-0.016, +0.030] | 3 / 2 |  |
| 1b | logistic regression − variety-only rule | 204 bands | 0.587 vs 0.676 | -0.089 [-0.120, -0.061] | single fits | ✗ | 0.619 vs 0.652 | -0.033 [-0.058, -0.008] | single fits | ✗ |
| 1b | CNN, mean pool − variety-only rule | 204 bands | 0.598 vs 0.676 | -0.079 [-0.098, -0.060] | 0 / 5 | ✗ | 0.635 vs 0.652 | -0.016 [-0.035, +0.002] | 2 / 3 |  |
| 1b | CNN, attention pool − variety-only rule | 204 bands | 0.595 vs 0.676 | -0.081 [-0.100, -0.060] | 0 / 5 | ✗ | 0.624 vs 0.652 | -0.028 [-0.049, -0.008] | 0 / 5 | ✗ |
| 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | 0.606 vs 0.676 | -0.071 [-0.090, -0.050] | 0 / 5 | ✗ | 0.636 vs 0.652 | -0.016 [-0.036, +0.004] | 0 / 5 |  |
| 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | 0.607 vs 0.676 | -0.069 [-0.087, -0.051] | 0 / 5 | ✗ | 0.633 vs 0.652 | -0.019 [-0.037, -0.001] | 0 / 5 | ✗ |
| 1b | CNN + CBAM, mean pool − variety-only rule | 204 bands | 0.612 vs 0.676 | -0.065 [-0.085, -0.045] | 0 / 5 | ✗ | 0.651 vs 0.652 | -0.001 [-0.022, +0.019] | 1 / 4 |  |
| 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | 0.626 vs 0.676 | -0.051 [-0.072, -0.029] | 0 / 5 | ✗ | 0.666 vs 0.652 | +0.014 [-0.004, +0.032] | 1 / 4 |  |
| 1b | small ResNet, mean pool − variety-only rule | 204 bands | 0.584 vs 0.676 | -0.092 [-0.114, -0.072] | 0 / 5 | ✗ | 0.632 vs 0.652 | -0.020 [-0.040, +0.000] | 0 / 5 |  |
| 1b | small ResNet, attention pool − variety-only rule | 204 bands | 0.611 vs 0.676 | -0.065 [-0.087, -0.041] | 0 / 5 | ✗ | 0.661 vs 0.652 | +0.009 [-0.012, +0.030] | 1 / 4 |  |
| 1b | small ResNet + ECA, mean pool − variety-only rule | 204 bands | 0.604 vs 0.676 | -0.072 [-0.092, -0.053] | 0 / 5 | ✗ | 0.654 vs 0.652 | +0.003 [-0.016, +0.022] | 2 / 3 |  |
| 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | 0.574 vs 0.676 | -0.103 [-0.122, -0.084] | 0 / 5 | ✗ | 0.622 vs 0.652 | -0.030 [-0.049, -0.011] | 0 / 5 | ✗ |
| 1b | small ResNet + CBAM, mean pool − variety-only rule | 204 bands | 0.612 vs 0.676 | -0.064 [-0.082, -0.047] | 0 / 5 | ✗ | 0.637 vs 0.652 | -0.015 [-0.031, +0.002] | 0 / 5 |  |
| 1b | small ResNet + CBAM, attention pool − variety-only rule | 204 bands | 0.585 vs 0.676 | -0.092 [-0.110, -0.074] | 0 / 5 | ✗ | 0.626 vs 0.652 | -0.026 [-0.044, -0.007] | 0 / 5 | ✗ |
| 1b | logistic regression − variety-only rule | RGB | 0.573 vs 0.676 | -0.103 [-0.130, -0.077] | single fits | ✗ | 0.622 vs 0.652 | -0.030 [-0.057, -0.005] | single fits | ✗ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.606 vs 0.598 | +0.008 [-0.009, +0.027] | 3 / 2 |  | 0.636 vs 0.635 | +0.000 [-0.011, +0.012] | 2 / 3 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.607 vs 0.595 | +0.012 [-0.010, +0.034] | 2 / 3 |  | 0.633 vs 0.624 | +0.010 [-0.004, +0.024] | 4 / 1 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.612 vs 0.598 | +0.014 [-0.007, +0.036] | 3 / 2 |  | 0.651 vs 0.635 | +0.016 [-0.000, +0.032] | 3 / 2 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.626 vs 0.595 | +0.030 [+0.007, +0.053] | 5 / 0 | ✓ | 0.666 vs 0.624 | +0.043 [+0.026, +0.060] | 5 / 0 | ✓ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.604 vs 0.584 | +0.020 [+0.002, +0.039] | 3 / 2 |  | 0.654 vs 0.632 | +0.023 [+0.011, +0.034] | 3 / 2 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.574 vs 0.611 | -0.038 [-0.059, -0.018] | 1 / 4 | ✗ | 0.622 vs 0.661 | -0.039 [-0.051, -0.026] | 1 / 4 | ✗ |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.612 vs 0.584 | +0.028 [+0.009, +0.047] | 3 / 2 |  | 0.637 vs 0.632 | +0.006 [-0.006, +0.018] | 3 / 2 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.585 vs 0.611 | -0.027 [-0.051, -0.005] | 0 / 5 | ✗ | 0.626 vs 0.661 | -0.035 [-0.049, -0.020] | 0 / 5 | ✗ |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.595 vs 0.598 | -0.002 [-0.023, +0.020] | 3 / 2 |  | 0.624 vs 0.635 | -0.012 [-0.027, +0.003] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.607 vs 0.606 | +0.001 [-0.019, +0.021] | 2 / 3 |  | 0.633 vs 0.636 | -0.002 [-0.015, +0.010] | 2 / 3 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.626 vs 0.612 | +0.014 [-0.008, +0.036] | 4 / 1 |  | 0.666 vs 0.651 | +0.015 [+0.001, +0.030] | 3 / 2 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.611 vs 0.584 | +0.027 [+0.002, +0.054] | 3 / 2 |  | 0.661 vs 0.632 | +0.029 [+0.010, +0.049] | 4 / 1 | ✓ |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.574 vs 0.604 | -0.031 [-0.053, -0.011] | 2 / 3 |  | 0.622 vs 0.654 | -0.032 [-0.051, -0.014] | 1 / 4 | ✗ |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.585 vs 0.612 | -0.028 [-0.049, -0.007] | 1 / 4 | ✗ | 0.626 vs 0.637 | -0.011 [-0.025, +0.003] | 1 / 4 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.584 vs 0.598 | -0.013 [-0.031, +0.004] | 3 / 2 |  | 0.632 vs 0.635 | -0.004 [-0.015, +0.007] | 2 / 3 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.611 vs 0.595 | +0.016 [-0.005, +0.038] | 4 / 1 |  | 0.661 vs 0.624 | +0.037 [+0.020, +0.054] | 5 / 0 | ✓ |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.604 vs 0.606 | -0.001 [-0.022, +0.019] | 3 / 2 |  | 0.654 vs 0.636 | +0.019 [+0.004, +0.033] | 4 / 1 | ✓ |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.574 vs 0.607 | -0.033 [-0.054, -0.013] | 3 / 2 |  | 0.622 vs 0.633 | -0.011 [-0.027, +0.005] | 2 / 3 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.612 vs 0.612 | +0.001 [-0.021, +0.022] | 3 / 2 |  | 0.637 vs 0.651 | -0.014 [-0.028, -0.000] | 3 / 2 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.585 vs 0.626 | -0.041 [-0.066, -0.018] | 1 / 4 | ✗ | 0.626 vs 0.666 | -0.040 [-0.056, -0.025] | 1 / 4 | ✗ |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.587 vs 0.573 | +0.014 [-0.021, +0.050] | single fits |  | 0.619 vs 0.622 | -0.003 [-0.033, +0.031] | single fits |  |

Marked, of 47 comparisons: balanced accuracy 2 ✓ and 18 ✗; AUC 6 ✓ and 10 ✗. Chance alone would put about 2.4 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, Baco

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.548 [0.489, 0.601] | 0.526 [0.453, 0.595] | 0.559 [0.507, 0.614] | 0.522 [0.458, 0.590] |
| CNN, mean pool | 0.500 [0.459, 0.543] | 0.497 [0.450, 0.544] | pending | pending |
| CNN, attention pool | 0.485 [0.441, 0.531] | 0.493 [0.446, 0.542] | pending | pending |
| CNN + ECA, mean pool | 0.483 [0.437, 0.533] | 0.519 [0.469, 0.569] | pending | pending |
| CNN + ECA, attention pool | 0.521 [0.486, 0.555] | 0.524 [0.482, 0.567] | pending | pending |
| CNN + CBAM, mean pool | 0.471 [0.424, 0.516] | 0.471 [0.427, 0.514] | pending | pending |
| CNN + CBAM, attention pool | 0.518 [0.466, 0.570] | 0.480 [0.420, 0.539] | pending | pending |
| small ResNet, mean pool | 0.508 [0.471, 0.545] | 0.483 [0.431, 0.539] | pending | pending |
| small ResNet, attention pool | 0.496 [0.446, 0.543] | 0.485 [0.430, 0.541] | pending | pending |
| small ResNet + ECA, mean pool | 0.509 [0.457, 0.559] | 0.522 [0.463, 0.584] | pending | pending |
| small ResNet + ECA, attention pool | 0.489 [0.440, 0.538] | 0.498 [0.444, 0.556] | pending | pending |
| small ResNet + CBAM, mean pool | 0.484 [0.450, 0.521] | 0.495 [0.453, 0.536] | pending | pending |
| small ResNet + CBAM, attention pool | 0.504 [0.456, 0.555] | 0.492 [0.438, 0.544] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.500 vs 0.548 | -0.048 [-0.120, +0.022] | 0 / 5 |  | 0.497 vs 0.526 | -0.029 [-0.099, +0.043] | 0 / 5 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.485 vs 0.548 | -0.062 [-0.128, +0.008] | 0 / 5 |  | 0.493 vs 0.526 | -0.033 [-0.105, +0.046] | 0 / 5 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.483 vs 0.548 | -0.065 [-0.137, +0.011] | 0 / 5 |  | 0.519 vs 0.526 | -0.007 [-0.083, +0.073] | 2 / 3 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.521 vs 0.548 | -0.027 [-0.090, +0.036] | 1 / 4 |  | 0.524 vs 0.526 | -0.002 [-0.077, +0.075] | 2 / 3 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.471 vs 0.548 | -0.077 [-0.141, -0.008] | 0 / 5 | ✗ | 0.471 vs 0.526 | -0.055 [-0.118, +0.014] | 0 / 5 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.518 vs 0.548 | -0.030 [-0.095, +0.037] | 0 / 5 |  | 0.480 vs 0.526 | -0.047 [-0.119, +0.029] | 1 / 4 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.508 vs 0.548 | -0.040 [-0.108, +0.031] | 0 / 5 |  | 0.483 vs 0.526 | -0.043 [-0.126, +0.043] | 0 / 5 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.496 vs 0.548 | -0.052 [-0.115, +0.012] | 0 / 5 |  | 0.485 vs 0.526 | -0.041 [-0.119, +0.041] | 1 / 4 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.509 vs 0.548 | -0.039 [-0.115, +0.036] | 0 / 5 |  | 0.522 vs 0.526 | -0.004 [-0.089, +0.086] | 3 / 2 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.489 vs 0.548 | -0.059 [-0.129, +0.015] | 1 / 4 |  | 0.498 vs 0.526 | -0.028 [-0.113, +0.061] | 2 / 3 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.484 vs 0.548 | -0.064 [-0.125, +0.003] | 0 / 5 |  | 0.495 vs 0.526 | -0.032 [-0.103, +0.045] | 1 / 4 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.504 vs 0.548 | -0.043 [-0.109, +0.029] | 0 / 5 |  | 0.492 vs 0.526 | -0.035 [-0.113, +0.046] | 2 / 3 |  |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.483 vs 0.500 | -0.017 [-0.067, +0.034] | 3 / 2 |  | 0.519 vs 0.497 | +0.023 [-0.017, +0.062] | 4 / 1 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.521 vs 0.485 | +0.036 [-0.017, +0.090] | 3 / 2 |  | 0.524 vs 0.493 | +0.031 [-0.005, +0.069] | 4 / 1 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.471 vs 0.500 | -0.029 [-0.086, +0.028] | 4 / 1 |  | 0.471 vs 0.497 | -0.026 [-0.075, +0.024] | 3 / 2 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.518 vs 0.485 | +0.033 [-0.023, +0.087] | 4 / 1 |  | 0.480 vs 0.493 | -0.013 [-0.060, +0.032] | 3 / 2 |  |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.509 vs 0.508 | +0.001 [-0.052, +0.053] | 3 / 2 |  | 0.522 vs 0.483 | +0.039 [-0.008, +0.086] | 4 / 1 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.489 vs 0.496 | -0.006 [-0.051, +0.041] | 2 / 3 |  | 0.498 vs 0.485 | +0.013 [-0.027, +0.050] | 3 / 2 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.484 vs 0.508 | -0.024 [-0.063, +0.015] | 3 / 2 |  | 0.495 vs 0.483 | +0.011 [-0.021, +0.048] | 4 / 1 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.504 vs 0.496 | +0.009 [-0.031, +0.051] | 3 / 2 |  | 0.492 vs 0.485 | +0.007 [-0.031, +0.044] | 3 / 2 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.485 vs 0.500 | -0.015 [-0.062, +0.035] | 4 / 1 |  | 0.493 vs 0.497 | -0.004 [-0.039, +0.032] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.521 vs 0.483 | +0.038 [-0.012, +0.088] | 3 / 2 |  | 0.524 vs 0.519 | +0.005 [-0.030, +0.041] | 2 / 3 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.518 vs 0.471 | +0.047 [-0.007, +0.101] | 2 / 3 |  | 0.480 vs 0.471 | +0.008 [-0.036, +0.055] | 2 / 3 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.496 vs 0.508 | -0.012 [-0.058, +0.035] | 4 / 1 |  | 0.485 vs 0.483 | +0.002 [-0.043, +0.047] | 4 / 1 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.489 vs 0.509 | -0.020 [-0.069, +0.028] | 2 / 3 |  | 0.498 vs 0.522 | -0.024 [-0.062, +0.015] | 3 / 2 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.504 vs 0.484 | +0.021 [-0.030, +0.068] | 3 / 2 |  | 0.492 vs 0.495 | -0.003 [-0.040, +0.032] | 2 / 3 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.508 vs 0.500 | +0.008 [-0.037, +0.054] | 3 / 2 |  | 0.483 vs 0.497 | -0.014 [-0.057, +0.029] | 4 / 1 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.496 vs 0.485 | +0.010 [-0.037, +0.056] | 3 / 2 |  | 0.485 vs 0.493 | -0.008 [-0.044, +0.029] | 2 / 3 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.509 vs 0.483 | +0.026 [-0.017, +0.070] | 2 / 3 |  | 0.522 vs 0.519 | +0.002 [-0.035, +0.043] | 3 / 2 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.489 vs 0.521 | -0.032 [-0.083, +0.017] | 3 / 2 |  | 0.498 vs 0.524 | -0.026 [-0.069, +0.016] | 3 / 2 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.484 vs 0.471 | +0.012 [-0.043, +0.073] | 3 / 2 |  | 0.495 vs 0.471 | +0.024 [-0.021, +0.071] | 3 / 2 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.504 vs 0.518 | -0.013 [-0.063, +0.038] | 3 / 2 |  | 0.492 vs 0.480 | +0.012 [-0.042, +0.067] | 3 / 2 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.548 vs 0.559 | -0.011 [-0.073, +0.047] | single fits |  | 0.526 vs 0.522 | +0.004 [-0.066, +0.081] | single fits |  |

Marked, of 33 comparisons: balanced accuracy 0 ✓ and 1 ✗; AUC 0 ✓ and 0 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, Chardonnay

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.597 [0.537, 0.656] | 0.566 [0.491, 0.637] | 0.634 [0.575, 0.694] | 0.656 [0.589, 0.722] |
| CNN, mean pool | 0.523 [0.474, 0.571] | 0.540 [0.482, 0.595] | pending | pending |
| CNN, attention pool | 0.512 [0.461, 0.563] | 0.536 [0.481, 0.590] | pending | pending |
| CNN + ECA, mean pool | 0.546 [0.502, 0.589] | 0.586 [0.533, 0.638] | pending | pending |
| CNN + ECA, attention pool | 0.572 [0.522, 0.622] | 0.539 [0.480, 0.594] | pending | pending |
| CNN + CBAM, mean pool | 0.560 [0.505, 0.614] | 0.577 [0.518, 0.636] | pending | pending |
| CNN + CBAM, attention pool | 0.591 [0.538, 0.642] | 0.584 [0.525, 0.642] | pending | pending |
| small ResNet, mean pool | 0.589 [0.537, 0.642] | 0.571 [0.515, 0.626] | pending | pending |
| small ResNet, attention pool | 0.542 [0.495, 0.589] | 0.543 [0.496, 0.589] | pending | pending |
| small ResNet + ECA, mean pool | 0.590 [0.543, 0.636] | 0.559 [0.504, 0.612] | pending | pending |
| small ResNet + ECA, attention pool | 0.595 [0.542, 0.644] | 0.564 [0.508, 0.618] | pending | pending |
| small ResNet + CBAM, mean pool | 0.572 [0.517, 0.624] | 0.559 [0.502, 0.613] | pending | pending |
| small ResNet + CBAM, attention pool | 0.554 [0.504, 0.605] | 0.541 [0.481, 0.597] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.523 vs 0.597 | -0.074 [-0.125, -0.022] | 0 / 5 | ✗ | 0.540 vs 0.566 | -0.026 [-0.065, +0.012] | 1 / 4 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.512 vs 0.597 | -0.084 [-0.133, -0.033] | 0 / 5 | ✗ | 0.536 vs 0.566 | -0.030 [-0.068, +0.008] | 0 / 5 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.546 vs 0.597 | -0.051 [-0.097, -0.001] | 0 / 5 | ✗ | 0.586 vs 0.566 | +0.020 [-0.021, +0.060] | 2 / 3 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.572 vs 0.597 | -0.024 [-0.077, +0.028] | 1 / 4 |  | 0.539 vs 0.566 | -0.027 [-0.068, +0.012] | 1 / 4 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.560 vs 0.597 | -0.037 [-0.081, +0.008] | 0 / 5 |  | 0.577 vs 0.566 | +0.011 [-0.034, +0.055] | 1 / 3 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.591 vs 0.597 | -0.005 [-0.058, +0.045] | 0 / 5 |  | 0.584 vs 0.566 | +0.018 [-0.031, +0.064] | 4 / 1 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.589 vs 0.597 | -0.008 [-0.061, +0.043] | 0 / 5 |  | 0.571 vs 0.566 | +0.005 [-0.038, +0.047] | 1 / 4 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.542 vs 0.597 | -0.055 [-0.100, -0.006] | 1 / 4 | ✗ | 0.543 vs 0.566 | -0.023 [-0.067, +0.022] | 1 / 4 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.590 vs 0.597 | -0.007 [-0.063, +0.046] | 0 / 5 |  | 0.559 vs 0.566 | -0.007 [-0.051, +0.038] | 1 / 4 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.595 vs 0.597 | -0.002 [-0.047, +0.043] | 0 / 5 |  | 0.564 vs 0.566 | -0.003 [-0.045, +0.039] | 1 / 4 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.572 vs 0.597 | -0.025 [-0.074, +0.027] | 2 / 3 |  | 0.559 vs 0.566 | -0.007 [-0.050, +0.033] | 2 / 3 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.554 vs 0.597 | -0.043 [-0.093, +0.008] | 0 / 5 |  | 0.541 vs 0.566 | -0.025 [-0.065, +0.011] | 2 / 3 |  |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.546 vs 0.523 | +0.023 [-0.015, +0.061] | 1 / 4 |  | 0.586 vs 0.540 | +0.046 [+0.025, +0.066] | 4 / 1 | ✓ |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.572 vs 0.512 | +0.060 [+0.017, +0.102] | 4 / 1 | ✓ | 0.539 vs 0.536 | +0.003 [-0.021, +0.025] | 3 / 2 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.560 vs 0.523 | +0.037 [-0.009, +0.083] | 2 / 3 |  | 0.577 vs 0.540 | +0.037 [+0.002, +0.070] | 4 / 1 | ✓ |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.591 vs 0.512 | +0.079 [+0.031, +0.127] | 4 / 1 | ✓ | 0.584 vs 0.536 | +0.048 [+0.014, +0.080] | 5 / 0 | ✓ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.590 vs 0.589 | +0.000 [-0.042, +0.041] | 2 / 3 |  | 0.559 vs 0.571 | -0.012 [-0.036, +0.011] | 1 / 4 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.595 vs 0.542 | +0.053 [+0.018, +0.089] | 1 / 4 |  | 0.564 vs 0.543 | +0.020 [-0.001, +0.041] | 4 / 1 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.572 vs 0.589 | -0.017 [-0.050, +0.015] | 3 / 2 |  | 0.559 vs 0.571 | -0.012 [-0.031, +0.007] | 3 / 2 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.554 vs 0.542 | +0.012 [-0.025, +0.050] | 3 / 2 |  | 0.541 vs 0.543 | -0.002 [-0.029, +0.023] | 3 / 2 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.512 vs 0.523 | -0.011 [-0.048, +0.025] | 1 / 4 |  | 0.536 vs 0.540 | -0.004 [-0.026, +0.016] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.572 vs 0.546 | +0.026 [-0.021, +0.070] | 5 / 0 |  | 0.539 vs 0.586 | -0.047 [-0.067, -0.028] | 0 / 5 | ✗ |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.591 vs 0.560 | +0.032 [-0.011, +0.071] | 2 / 3 |  | 0.584 vs 0.577 | +0.007 [-0.024, +0.038] | 4 / 1 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.542 vs 0.589 | -0.047 [-0.089, -0.006] | 2 / 3 |  | 0.543 vs 0.571 | -0.028 [-0.056, +0.001] | 2 / 3 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.595 vs 0.590 | +0.006 [-0.036, +0.044] | 3 / 2 |  | 0.564 vs 0.559 | +0.004 [-0.013, +0.022] | 4 / 1 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.554 vs 0.572 | -0.018 [-0.056, +0.023] | 2 / 3 |  | 0.541 vs 0.559 | -0.018 [-0.038, +0.003] | 2 / 3 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.589 vs 0.523 | +0.066 [+0.025, +0.110] | 3 / 2 |  | 0.571 vs 0.540 | +0.030 [+0.008, +0.054] | 5 / 0 | ✓ |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.542 vs 0.512 | +0.030 [-0.002, +0.059] | 4 / 1 |  | 0.543 vs 0.536 | +0.007 [-0.013, +0.028] | 3 / 2 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.590 vs 0.546 | +0.044 [-0.000, +0.085] | 3 / 2 |  | 0.559 vs 0.586 | -0.027 [-0.049, -0.006] | 1 / 4 | ✗ |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.595 vs 0.572 | +0.023 [-0.014, +0.060] | 3 / 2 |  | 0.564 vs 0.539 | +0.024 [+0.004, +0.045] | 4 / 1 | ✓ |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.572 vs 0.560 | +0.012 [-0.032, +0.056] | 3 / 2 |  | 0.559 vs 0.577 | -0.018 [-0.055, +0.017] | 2 / 3 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.554 vs 0.591 | -0.037 [-0.080, +0.008] | 1 / 4 |  | 0.541 vs 0.584 | -0.043 [-0.073, -0.013] | 1 / 4 | ✗ |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.597 vs 0.634 | -0.037 [-0.091, +0.016] | single fits |  | 0.566 vs 0.656 | -0.090 [-0.149, -0.030] | single fits | ✗ |

Marked, of 33 comparisons: balanced accuracy 2 ✓ and 4 ✗; AUC 5 ✓ and 4 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, Vidal

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.538 [0.476, 0.601] | 0.566 [0.492, 0.636] | 0.528 [0.461, 0.589] | 0.607 [0.533, 0.674] |
| CNN, mean pool | 0.518 [0.467, 0.570] | 0.542 [0.483, 0.599] | pending | pending |
| CNN, attention pool | 0.493 [0.441, 0.541] | 0.538 [0.484, 0.589] | pending | pending |
| CNN + ECA, mean pool | 0.486 [0.443, 0.531] | 0.559 [0.515, 0.602] | pending | pending |
| CNN + ECA, attention pool | 0.510 [0.466, 0.554] | 0.535 [0.483, 0.583] | pending | pending |
| CNN + CBAM, mean pool | 0.549 [0.503, 0.594] | 0.574 [0.521, 0.623] | pending | pending |
| CNN + CBAM, attention pool | 0.480 [0.437, 0.526] | 0.519 [0.475, 0.560] | pending | pending |
| small ResNet, mean pool | 0.546 [0.490, 0.600] | 0.577 [0.517, 0.635] | pending | pending |
| small ResNet, attention pool | 0.495 [0.445, 0.544] | 0.529 [0.472, 0.584] | pending | pending |
| small ResNet + ECA, mean pool | 0.561 [0.509, 0.612] | 0.587 [0.532, 0.644] | pending | pending |
| small ResNet + ECA, attention pool | 0.500 [0.456, 0.547] | 0.524 [0.477, 0.568] | pending | pending |
| small ResNet + CBAM, mean pool | 0.528 [0.480, 0.575] | 0.543 [0.494, 0.592] | pending | pending |
| small ResNet + CBAM, attention pool | 0.510 [0.473, 0.547] | 0.489 [0.444, 0.528] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.518 vs 0.538 | -0.020 [-0.091, +0.048] | 1 / 4 |  | 0.542 vs 0.566 | -0.024 [-0.091, +0.043] | 2 / 3 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.493 vs 0.538 | -0.045 [-0.109, +0.018] | 1 / 4 |  | 0.538 vs 0.566 | -0.027 [-0.088, +0.032] | 2 / 3 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.486 vs 0.538 | -0.052 [-0.117, +0.010] | 1 / 4 |  | 0.559 vs 0.566 | -0.007 [-0.067, +0.051] | 1 / 4 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.510 vs 0.538 | -0.028 [-0.093, +0.037] | 0 / 5 |  | 0.535 vs 0.566 | -0.031 [-0.090, +0.029] | 0 / 5 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.549 vs 0.538 | +0.011 [-0.046, +0.071] | 1 / 4 |  | 0.574 vs 0.566 | +0.008 [-0.051, +0.069] | 2 / 3 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.480 vs 0.538 | -0.058 [-0.121, +0.001] | 0 / 5 |  | 0.519 vs 0.566 | -0.047 [-0.107, +0.013] | 0 / 5 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.546 vs 0.538 | +0.008 [-0.058, +0.074] | 2 / 3 |  | 0.577 vs 0.566 | +0.011 [-0.052, +0.077] | 3 / 2 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.495 vs 0.538 | -0.043 [-0.110, +0.025] | 0 / 5 |  | 0.529 vs 0.566 | -0.037 [-0.099, +0.028] | 0 / 5 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.561 vs 0.538 | +0.023 [-0.041, +0.087] | 3 / 1 |  | 0.587 vs 0.566 | +0.021 [-0.040, +0.084] | 3 / 2 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.500 vs 0.538 | -0.038 [-0.100, +0.024] | 0 / 5 |  | 0.524 vs 0.566 | -0.042 [-0.100, +0.015] | 1 / 4 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.528 vs 0.538 | -0.009 [-0.075, +0.053] | 1 / 4 |  | 0.543 vs 0.566 | -0.023 [-0.081, +0.035] | 1 / 4 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.510 vs 0.538 | -0.028 [-0.092, +0.034] | 0 / 5 |  | 0.489 vs 0.566 | -0.077 [-0.135, -0.021] | 0 / 5 | ✗ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.486 vs 0.518 | -0.032 [-0.074, +0.008] | 3 / 2 |  | 0.559 vs 0.542 | +0.016 [-0.009, +0.042] | 3 / 2 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.510 vs 0.493 | +0.017 [-0.021, +0.053] | 3 / 2 |  | 0.535 vs 0.538 | -0.003 [-0.029, +0.023] | 3 / 2 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.549 vs 0.518 | +0.032 [-0.018, +0.080] | 2 / 3 |  | 0.574 vs 0.542 | +0.031 [-0.008, +0.070] | 3 / 2 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.480 vs 0.493 | -0.013 [-0.046, +0.019] | 3 / 2 |  | 0.519 vs 0.538 | -0.019 [-0.047, +0.007] | 2 / 3 |  |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.561 vs 0.546 | +0.015 [-0.016, +0.049] | 3 / 2 |  | 0.587 vs 0.577 | +0.010 [-0.014, +0.036] | 4 / 1 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.500 vs 0.495 | +0.005 [-0.026, +0.037] | 1 / 4 |  | 0.524 vs 0.529 | -0.005 [-0.030, +0.019] | 2 / 3 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.528 vs 0.546 | -0.017 [-0.061, +0.025] | 2 / 3 |  | 0.543 vs 0.577 | -0.034 [-0.067, +0.000] | 1 / 4 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.510 vs 0.495 | +0.015 [-0.028, +0.058] | 0 / 5 |  | 0.489 vs 0.529 | -0.040 [-0.076, -0.004] | 1 / 4 | ✗ |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.493 vs 0.518 | -0.025 [-0.069, +0.017] | 3 / 2 |  | 0.538 vs 0.542 | -0.004 [-0.030, +0.023] | 3 / 2 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.510 vs 0.486 | +0.024 [-0.006, +0.056] | 1 / 4 |  | 0.535 vs 0.559 | -0.023 [-0.051, +0.002] | 1 / 4 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.480 vs 0.549 | -0.069 [-0.117, -0.022] | 0 / 5 | ✗ | 0.519 vs 0.574 | -0.055 [-0.084, -0.026] | 0 / 5 | ✗ |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.495 vs 0.546 | -0.051 [-0.095, -0.003] | 2 / 3 |  | 0.529 vs 0.577 | -0.048 [-0.079, -0.018] | 0 / 5 | ✗ |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.500 vs 0.561 | -0.061 [-0.100, -0.024] | 1 / 4 | ✗ | 0.524 vs 0.587 | -0.063 [-0.094, -0.035] | 0 / 5 | ✗ |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.510 vs 0.528 | -0.018 [-0.059, +0.025] | 2 / 3 |  | 0.489 vs 0.543 | -0.055 [-0.084, -0.024] | 0 / 5 | ✗ |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.546 vs 0.518 | +0.028 [-0.016, +0.072] | 3 / 2 |  | 0.577 vs 0.542 | +0.035 [+0.003, +0.067] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.495 vs 0.493 | +0.002 [-0.038, +0.043] | 3 / 2 |  | 0.529 vs 0.538 | -0.010 [-0.037, +0.017] | 2 / 3 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.561 vs 0.486 | +0.075 [+0.032, +0.122] | 4 / 1 | ✓ | 0.587 vs 0.559 | +0.029 [-0.002, +0.058] | 4 / 1 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.500 vs 0.510 | -0.009 [-0.046, +0.027] | 2 / 3 |  | 0.524 vs 0.535 | -0.011 [-0.036, +0.011] | 1 / 4 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.528 vs 0.549 | -0.021 [-0.066, +0.022] | 2 / 3 |  | 0.543 vs 0.574 | -0.030 [-0.060, -0.002] | 1 / 4 | ✗ |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.510 vs 0.480 | +0.030 [-0.010, +0.071] | 3 / 2 |  | 0.489 vs 0.519 | -0.030 [-0.057, -0.002] | 2 / 3 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.538 vs 0.528 | +0.010 [-0.050, +0.074] | single fits |  | 0.566 vs 0.607 | -0.041 [-0.110, +0.034] | single fits |  |

Marked, of 33 comparisons: balanced accuracy 1 ✓ and 2 ✗; AUC 0 ✓ and 7 ✗. Chance alone would put about 1.6 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, pooled, all 998 leaves

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.663 [0.643, 0.684] | 0.724 [0.705, 0.742] | 0.641 [0.619, 0.664] | 0.683 [0.660, 0.706] |
| variety-only rule | 0.706 [0.706, 0.706] | 0.764 [0.764, 0.764] | 0.706 [0.706, 0.706] | 0.764 [0.764, 0.764] |
| CNN, mean pool | 0.693 [0.674, 0.710] | 0.748 [0.732, 0.764] | pending | pending |
| CNN, attention pool | 0.677 [0.660, 0.694] | 0.748 [0.732, 0.763] | pending | pending |
| CNN + ECA, mean pool | 0.685 [0.670, 0.701] | 0.752 [0.737, 0.766] | pending | pending |
| CNN + ECA, attention pool | 0.683 [0.665, 0.701] | 0.741 [0.726, 0.756] | pending | pending |
| CNN + CBAM, mean pool | 0.702 [0.684, 0.720] | 0.763 [0.748, 0.778] | pending | pending |
| CNN + CBAM, attention pool | 0.699 [0.679, 0.719] | 0.763 [0.748, 0.778] | pending | pending |
| small ResNet, mean pool | 0.691 [0.673, 0.709] | 0.755 [0.739, 0.771] | pending | pending |
| small ResNet, attention pool | 0.686 [0.667, 0.704] | 0.752 [0.737, 0.767] | pending | pending |
| small ResNet + ECA, mean pool | 0.691 [0.672, 0.708] | 0.755 [0.738, 0.771] | pending | pending |
| small ResNet + ECA, attention pool | 0.668 [0.651, 0.687] | 0.747 [0.731, 0.762] | pending | pending |
| small ResNet + CBAM, mean pool | 0.690 [0.671, 0.708] | 0.754 [0.738, 0.769] | pending | pending |
| small ResNet + CBAM, attention pool | 0.689 [0.670, 0.708] | 0.760 [0.746, 0.774] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **1b** against the variety-only rule (pooled sets); **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.693 vs 0.663 | +0.030 [+0.007, +0.052] | 5 / 0 | ✓ | 0.748 vs 0.724 | +0.025 [+0.011, +0.039] | 5 / 0 | ✓ |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.677 vs 0.663 | +0.014 [-0.009, +0.036] | 3 / 2 |  | 0.748 vs 0.724 | +0.024 [+0.009, +0.039] | 5 / 0 | ✓ |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.685 vs 0.663 | +0.022 [-0.000, +0.042] | 2 / 3 |  | 0.752 vs 0.724 | +0.028 [+0.014, +0.043] | 3 / 2 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.683 vs 0.663 | +0.020 [-0.003, +0.042] | 2 / 3 |  | 0.741 vs 0.724 | +0.017 [+0.004, +0.031] | 3 / 2 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.702 vs 0.663 | +0.039 [+0.016, +0.062] | 5 / 0 | ✓ | 0.763 vs 0.724 | +0.040 [+0.024, +0.056] | 5 / 0 | ✓ |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.699 vs 0.663 | +0.036 [+0.013, +0.057] | 5 / 0 | ✓ | 0.763 vs 0.724 | +0.039 [+0.025, +0.055] | 5 / 0 | ✓ |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.691 vs 0.663 | +0.028 [+0.005, +0.051] | 5 / 0 | ✓ | 0.755 vs 0.724 | +0.031 [+0.018, +0.045] | 5 / 0 | ✓ |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.686 vs 0.663 | +0.023 [-0.000, +0.045] | 2 / 3 |  | 0.752 vs 0.724 | +0.028 [+0.013, +0.043] | 3 / 2 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.691 vs 0.663 | +0.027 [+0.003, +0.051] | 4 / 1 | ✓ | 0.755 vs 0.724 | +0.031 [+0.017, +0.046] | 4 / 1 | ✓ |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.668 vs 0.663 | +0.005 [-0.017, +0.027] | 3 / 2 |  | 0.747 vs 0.724 | +0.024 [+0.009, +0.037] | 4 / 1 | ✓ |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.690 vs 0.663 | +0.026 [+0.003, +0.048] | 4 / 1 | ✓ | 0.754 vs 0.724 | +0.031 [+0.016, +0.046] | 5 / 0 | ✓ |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.689 vs 0.663 | +0.026 [+0.001, +0.049] | 4 / 1 | ✓ | 0.760 vs 0.724 | +0.037 [+0.022, +0.051] | 5 / 0 | ✓ |
| 1b | logistic regression − variety-only rule | 204 bands | 0.663 vs 0.706 | -0.043 [-0.063, -0.022] | single fits | ✗ | 0.724 vs 0.764 | -0.041 [-0.059, -0.022] | single fits | ✗ |
| 1b | CNN, mean pool − variety-only rule | 204 bands | 0.693 vs 0.706 | -0.013 [-0.032, +0.004] | 0 / 5 |  | 0.748 vs 0.764 | -0.016 [-0.033, +0.000] | 0 / 5 |  |
| 1b | CNN, attention pool − variety-only rule | 204 bands | 0.677 vs 0.706 | -0.028 [-0.046, -0.012] | 0 / 5 | ✗ | 0.748 vs 0.764 | -0.016 [-0.032, -0.001] | 0 / 5 | ✗ |
| 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | 0.685 vs 0.706 | -0.021 [-0.036, -0.005] | 0 / 5 | ✗ | 0.752 vs 0.764 | -0.012 [-0.027, +0.002] | 0 / 5 |  |
| 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | 0.683 vs 0.706 | -0.023 [-0.041, -0.005] | 0 / 5 | ✗ | 0.741 vs 0.764 | -0.023 [-0.038, -0.008] | 0 / 5 | ✗ |
| 1b | CNN + CBAM, mean pool − variety-only rule | 204 bands | 0.702 vs 0.706 | -0.004 [-0.022, +0.014] | 0 / 5 |  | 0.763 vs 0.764 | -0.001 [-0.016, +0.014] | 0 / 5 |  |
| 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | 0.699 vs 0.706 | -0.007 [-0.027, +0.013] | 0 / 5 |  | 0.763 vs 0.764 | -0.001 [-0.016, +0.013] | 0 / 5 |  |
| 1b | small ResNet, mean pool − variety-only rule | 204 bands | 0.691 vs 0.706 | -0.015 [-0.033, +0.003] | 0 / 5 |  | 0.755 vs 0.764 | -0.009 [-0.026, +0.007] | 0 / 5 |  |
| 1b | small ResNet, attention pool − variety-only rule | 204 bands | 0.686 vs 0.706 | -0.020 [-0.039, -0.002] | 0 / 5 | ✗ | 0.752 vs 0.764 | -0.012 [-0.027, +0.002] | 0 / 5 |  |
| 1b | small ResNet + ECA, mean pool − variety-only rule | 204 bands | 0.691 vs 0.706 | -0.015 [-0.034, +0.002] | 0 / 5 |  | 0.755 vs 0.764 | -0.009 [-0.026, +0.007] | 0 / 5 |  |
| 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | 0.668 vs 0.706 | -0.037 [-0.055, -0.019] | 0 / 5 | ✗ | 0.747 vs 0.764 | -0.017 [-0.033, -0.002] | 0 / 5 | ✗ |
| 1b | small ResNet + CBAM, mean pool − variety-only rule | 204 bands | 0.690 vs 0.706 | -0.016 [-0.035, +0.002] | 0 / 5 |  | 0.754 vs 0.764 | -0.010 [-0.026, +0.005] | 0 / 5 |  |
| 1b | small ResNet + CBAM, attention pool − variety-only rule | 204 bands | 0.689 vs 0.706 | -0.017 [-0.036, +0.002] | 0 / 5 |  | 0.760 vs 0.764 | -0.004 [-0.018, +0.010] | 0 / 5 |  |
| 1b | logistic regression − variety-only rule | RGB | 0.641 vs 0.706 | -0.065 [-0.087, -0.042] | single fits | ✗ | 0.683 vs 0.764 | -0.081 [-0.104, -0.058] | single fits | ✗ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.685 vs 0.693 | -0.008 [-0.020, +0.005] | 1 / 4 |  | 0.752 vs 0.748 | +0.004 [-0.003, +0.011] | 2 / 3 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.683 vs 0.677 | +0.006 [-0.010, +0.022] | 2 / 3 |  | 0.741 vs 0.748 | -0.007 [-0.015, +0.001] | 2 / 3 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.702 vs 0.693 | +0.009 [-0.008, +0.027] | 3 / 2 |  | 0.763 vs 0.748 | +0.015 [+0.005, +0.025] | 4 / 1 | ✓ |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.699 vs 0.677 | +0.021 [+0.004, +0.040] | 4 / 1 | ✓ | 0.763 vs 0.748 | +0.015 [+0.005, +0.026] | 4 / 1 | ✓ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.691 vs 0.691 | -0.001 [-0.016, +0.014] | 3 / 2 |  | 0.755 vs 0.755 | -0.000 [-0.007, +0.006] | 3 / 2 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.668 vs 0.686 | -0.018 [-0.034, -0.001] | 3 / 2 |  | 0.747 vs 0.752 | -0.005 [-0.012, +0.003] | 2 / 3 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.690 vs 0.691 | -0.002 [-0.016, +0.013] | 0 / 5 |  | 0.754 vs 0.755 | -0.001 [-0.008, +0.007] | 2 / 3 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.689 vs 0.686 | +0.003 [-0.015, +0.021] | 4 / 1 |  | 0.760 vs 0.752 | +0.008 [+0.001, +0.016] | 4 / 1 | ✓ |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.677 vs 0.693 | -0.015 [-0.029, -0.002] | 0 / 5 | ✗ | 0.748 vs 0.748 | -0.000 [-0.008, +0.007] | 2 / 3 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.683 vs 0.685 | -0.002 [-0.016, +0.012] | 2 / 3 |  | 0.741 vs 0.752 | -0.011 [-0.017, -0.004] | 2 / 3 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.699 vs 0.702 | -0.003 [-0.020, +0.014] | 2 / 3 |  | 0.763 vs 0.763 | -0.000 [-0.010, +0.008] | 3 / 2 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.686 vs 0.691 | -0.005 [-0.021, +0.011] | 2 / 3 |  | 0.752 vs 0.755 | -0.003 [-0.011, +0.005] | 2 / 3 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.668 vs 0.691 | -0.022 [-0.039, -0.006] | 0 / 5 | ✗ | 0.747 vs 0.755 | -0.007 [-0.015, +0.001] | 2 / 3 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.689 vs 0.690 | -0.000 [-0.017, +0.017] | 3 / 2 |  | 0.760 vs 0.754 | +0.006 [-0.001, +0.013] | 3 / 2 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.691 vs 0.693 | -0.002 [-0.015, +0.012] | 4 / 1 |  | 0.755 vs 0.748 | +0.007 [+0.001, +0.012] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.686 vs 0.677 | +0.009 [-0.006, +0.024] | 2 / 3 |  | 0.752 vs 0.748 | +0.004 [-0.004, +0.012] | 2 / 3 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.691 vs 0.685 | +0.005 [-0.009, +0.020] | 5 / 0 |  | 0.755 vs 0.752 | +0.003 [-0.005, +0.010] | 5 / 0 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.668 vs 0.683 | -0.015 [-0.032, +0.002] | 3 / 2 |  | 0.747 vs 0.741 | +0.006 [-0.002, +0.015] | 4 / 1 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.690 vs 0.702 | -0.013 [-0.030, +0.006] | 2 / 3 |  | 0.754 vs 0.763 | -0.009 [-0.019, +0.001] | 1 / 4 |  |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.689 vs 0.699 | -0.010 [-0.028, +0.008] | 1 / 4 |  | 0.760 vs 0.763 | -0.003 [-0.012, +0.006] | 2 / 3 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.663 vs 0.641 | +0.022 [-0.004, +0.049] | single fits |  | 0.724 vs 0.683 | +0.041 [+0.019, +0.064] | single fits | ✓ |

Marked, of 47 comparisons: balanced accuracy 8 ✓ and 9 ✗; AUC 13 ✓ and 5 ✗. Chance alone would put about 2.4 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, pooled, trained without Cab Franc (Baco + Chardonnay + Vidal)

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.580 [0.557, 0.605] | 0.638 [0.614, 0.662] | 0.610 [0.588, 0.633] | 0.651 [0.628, 0.672] |
| variety-only rule | 0.619 [0.619, 0.619] | 0.650 [0.650, 0.650] | 0.619 [0.619, 0.619] | 0.650 [0.650, 0.650] |
| CNN, mean pool | 0.604 [0.582, 0.626] | 0.642 [0.620, 0.663] | pending | pending |
| CNN, attention pool | 0.608 [0.585, 0.631] | 0.649 [0.628, 0.670] | pending | pending |
| CNN + ECA, mean pool | 0.596 [0.574, 0.620] | 0.622 [0.602, 0.642] | pending | pending |
| CNN + ECA, attention pool | 0.625 [0.605, 0.643] | 0.647 [0.627, 0.668] | pending | pending |
| CNN + CBAM, mean pool | 0.607 [0.586, 0.628] | 0.635 [0.613, 0.657] | pending | pending |
| CNN + CBAM, attention pool | 0.590 [0.567, 0.612] | 0.633 [0.611, 0.655] | pending | pending |
| small ResNet, mean pool | 0.604 [0.580, 0.626] | 0.651 [0.628, 0.673] | pending | pending |
| small ResNet, attention pool | 0.587 [0.564, 0.609] | 0.641 [0.621, 0.662] | pending | pending |
| small ResNet + ECA, mean pool | 0.616 [0.593, 0.638] | 0.652 [0.629, 0.675] | pending | pending |
| small ResNet + ECA, attention pool | 0.615 [0.592, 0.637] | 0.652 [0.630, 0.674] | pending | pending |
| small ResNet + CBAM, mean pool | 0.609 [0.587, 0.630] | 0.648 [0.626, 0.671] | pending | pending |
| small ResNet + CBAM, attention pool | 0.617 [0.592, 0.643] | 0.651 [0.629, 0.672] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **1b** against the variety-only rule (pooled sets); **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.604 vs 0.580 | +0.025 [-0.000, +0.049] | 4 / 1 |  | 0.642 vs 0.638 | +0.004 [-0.013, +0.020] | 2 / 3 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.608 vs 0.580 | +0.029 [+0.004, +0.051] | 5 / 0 | ✓ | 0.649 vs 0.638 | +0.011 [-0.005, +0.027] | 2 / 3 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.596 vs 0.580 | +0.017 [-0.006, +0.039] | 5 / 0 |  | 0.622 vs 0.638 | -0.016 [-0.032, +0.001] | 0 / 5 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.625 vs 0.580 | +0.045 [+0.020, +0.068] | 4 / 1 | ✓ | 0.647 vs 0.638 | +0.009 [-0.008, +0.026] | 2 / 3 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.607 vs 0.580 | +0.028 [-0.000, +0.054] | 4 / 1 |  | 0.635 vs 0.638 | -0.003 [-0.022, +0.016] | 2 / 3 |  |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.590 vs 0.580 | +0.010 [-0.013, +0.032] | 3 / 2 |  | 0.633 vs 0.638 | -0.005 [-0.022, +0.012] | 1 / 4 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.604 vs 0.580 | +0.024 [-0.002, +0.050] | 3 / 2 |  | 0.651 vs 0.638 | +0.012 [-0.005, +0.029] | 3 / 2 |  |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.587 vs 0.580 | +0.007 [-0.019, +0.032] | 4 / 1 |  | 0.641 vs 0.638 | +0.003 [-0.013, +0.021] | 2 / 3 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.616 vs 0.580 | +0.036 [+0.008, +0.063] | 5 / 0 | ✓ | 0.652 vs 0.638 | +0.014 [-0.005, +0.033] | 4 / 1 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.615 vs 0.580 | +0.035 [+0.007, +0.062] | 5 / 0 | ✓ | 0.652 vs 0.638 | +0.014 [-0.005, +0.032] | 2 / 3 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.609 vs 0.580 | +0.029 [+0.003, +0.055] | 5 / 0 | ✓ | 0.648 vs 0.638 | +0.010 [-0.008, +0.029] | 3 / 2 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.617 vs 0.580 | +0.037 [+0.009, +0.064] | 5 / 0 | ✓ | 0.651 vs 0.638 | +0.013 [-0.006, +0.030] | 4 / 1 |  |
| 1b | logistic regression − variety-only rule | 204 bands | 0.580 vs 0.619 | -0.039 [-0.062, -0.014] | single fits | ✗ | 0.638 vs 0.650 | -0.012 [-0.035, +0.013] | single fits |  |
| 1b | CNN, mean pool − variety-only rule | 204 bands | 0.604 vs 0.619 | -0.015 [-0.037, +0.007] | 0 / 5 |  | 0.642 vs 0.650 | -0.008 [-0.030, +0.013] | 0 / 5 |  |
| 1b | CNN, attention pool − variety-only rule | 204 bands | 0.608 vs 0.619 | -0.011 [-0.034, +0.012] | 0 / 5 |  | 0.649 vs 0.650 | -0.000 [-0.021, +0.021] | 1 / 4 |  |
| 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | 0.596 vs 0.619 | -0.023 [-0.045, +0.001] | 0 / 5 |  | 0.622 vs 0.650 | -0.027 [-0.048, -0.007] | 0 / 5 | ✗ |
| 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | 0.625 vs 0.619 | +0.006 [-0.014, +0.024] | 0 / 5 |  | 0.647 vs 0.650 | -0.002 [-0.023, +0.018] | 0 / 5 |  |
| 1b | CNN + CBAM, mean pool − variety-only rule | 204 bands | 0.607 vs 0.619 | -0.012 [-0.033, +0.009] | 0 / 5 |  | 0.635 vs 0.650 | -0.015 [-0.037, +0.007] | 2 / 3 |  |
| 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | 0.590 vs 0.619 | -0.029 [-0.052, -0.007] | 1 / 4 | ✗ | 0.633 vs 0.650 | -0.017 [-0.039, +0.005] | 1 / 4 |  |
| 1b | small ResNet, mean pool − variety-only rule | 204 bands | 0.604 vs 0.619 | -0.015 [-0.039, +0.007] | 1 / 4 |  | 0.651 vs 0.650 | +0.001 [-0.021, +0.024] | 2 / 3 |  |
| 1b | small ResNet, attention pool − variety-only rule | 204 bands | 0.587 vs 0.619 | -0.032 [-0.055, -0.010] | 0 / 5 | ✗ | 0.641 vs 0.650 | -0.009 [-0.029, +0.012] | 1 / 4 |  |
| 1b | small ResNet + ECA, mean pool − variety-only rule | 204 bands | 0.616 vs 0.619 | -0.003 [-0.026, +0.019] | 0 / 5 |  | 0.652 vs 0.650 | +0.002 [-0.021, +0.025] | 1 / 4 |  |
| 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | 0.615 vs 0.619 | -0.004 [-0.027, +0.018] | 2 / 3 |  | 0.652 vs 0.650 | +0.002 [-0.020, +0.024] | 2 / 3 |  |
| 1b | small ResNet + CBAM, mean pool − variety-only rule | 204 bands | 0.609 vs 0.619 | -0.010 [-0.032, +0.011] | 1 / 4 |  | 0.648 vs 0.650 | -0.001 [-0.023, +0.021] | 2 / 3 |  |
| 1b | small ResNet + CBAM, attention pool − variety-only rule | 204 bands | 0.617 vs 0.619 | -0.002 [-0.027, +0.024] | 1 / 4 |  | 0.651 vs 0.650 | +0.001 [-0.021, +0.023] | 1 / 4 |  |
| 1b | logistic regression − variety-only rule | RGB | 0.610 vs 0.619 | -0.009 [-0.031, +0.014] | single fits |  | 0.651 vs 0.650 | +0.001 [-0.022, +0.022] | single fits |  |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.596 vs 0.604 | -0.008 [-0.027, +0.012] | 2 / 3 |  | 0.622 vs 0.642 | -0.019 [-0.028, -0.011] | 0 / 5 | ✗ |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.625 vs 0.608 | +0.016 [-0.004, +0.037] | 3 / 2 |  | 0.647 vs 0.649 | -0.002 [-0.012, +0.008] | 3 / 2 |  |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.607 vs 0.604 | +0.003 [-0.018, +0.024] | 2 / 3 |  | 0.635 vs 0.642 | -0.007 [-0.018, +0.004] | 2 / 3 |  |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.590 vs 0.608 | -0.019 [-0.037, +0.001] | 1 / 4 |  | 0.633 vs 0.649 | -0.016 [-0.026, -0.007] | 1 / 4 | ✗ |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.616 vs 0.604 | +0.013 [-0.005, +0.030] | 2 / 3 |  | 0.652 vs 0.651 | +0.001 [-0.007, +0.010] | 3 / 2 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.615 vs 0.587 | +0.028 [+0.010, +0.048] | 4 / 1 | ✓ | 0.652 vs 0.641 | +0.010 [+0.002, +0.019] | 4 / 1 | ✓ |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.609 vs 0.604 | +0.005 [-0.012, +0.022] | 2 / 3 |  | 0.648 vs 0.651 | -0.002 [-0.010, +0.005] | 2 / 3 |  |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.617 vs 0.587 | +0.030 [+0.012, +0.050] | 4 / 1 | ✓ | 0.651 vs 0.641 | +0.010 [+0.002, +0.017] | 3 / 2 |  |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.608 vs 0.604 | +0.004 [-0.011, +0.018] | 2 / 3 |  | 0.649 vs 0.642 | +0.008 [-0.001, +0.017] | 3 / 2 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.625 vs 0.596 | +0.028 [+0.008, +0.048] | 4 / 1 | ✓ | 0.647 vs 0.622 | +0.025 [+0.017, +0.034] | 5 / 0 | ✓ |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.590 vs 0.607 | -0.018 [-0.040, +0.003] | 3 / 2 |  | 0.633 vs 0.635 | -0.002 [-0.012, +0.008] | 4 / 1 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.587 vs 0.604 | -0.017 [-0.036, +0.001] | 3 / 2 |  | 0.641 vs 0.651 | -0.009 [-0.018, -0.000] | 3 / 2 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.615 vs 0.616 | -0.001 [-0.018, +0.015] | 3 / 2 |  | 0.652 vs 0.652 | -0.000 [-0.010, +0.010] | 2 / 3 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.617 vs 0.609 | +0.008 [-0.010, +0.027] | 2 / 3 |  | 0.651 vs 0.648 | +0.002 [-0.004, +0.010] | 2 / 3 |  |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.604 vs 0.604 | -0.001 [-0.019, +0.017] | 3 / 2 |  | 0.651 vs 0.642 | +0.009 [-0.000, +0.017] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.587 vs 0.608 | -0.021 [-0.042, -0.001] | 1 / 4 | ✗ | 0.641 vs 0.649 | -0.008 [-0.019, +0.002] | 3 / 2 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.616 vs 0.596 | +0.020 [-0.002, +0.041] | 4 / 1 |  | 0.652 vs 0.622 | +0.030 [+0.017, +0.042] | 5 / 0 | ✓ |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.615 vs 0.625 | -0.010 [-0.027, +0.008] | 4 / 1 |  | 0.652 vs 0.647 | +0.004 [-0.005, +0.014] | 3 / 2 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.609 vs 0.607 | +0.001 [-0.019, +0.022] | 4 / 1 |  | 0.648 vs 0.635 | +0.013 [+0.004, +0.023] | 4 / 1 | ✓ |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.617 vs 0.590 | +0.027 [+0.005, +0.049] | 4 / 1 | ✓ | 0.651 vs 0.633 | +0.018 [+0.009, +0.028] | 4 / 1 | ✓ |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.580 vs 0.610 | -0.030 [-0.058, -0.001] | single fits | ✗ | 0.638 vs 0.651 | -0.013 [-0.037, +0.012] | single fits |  |

Marked, of 47 comparisons: balanced accuracy 10 ✓ and 5 ✗; AUC 5 ✓ and 3 ✗. Chance alone would put about 2.4 intervals per score off 0 (fewer once the seeds must agree).

### Red blotch, pooled, all 998 leaves, scored without Cab Franc (same models as all 998)

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.590 [0.566, 0.615] | 0.634 [0.611, 0.658] | 0.585 [0.559, 0.611] | 0.615 [0.587, 0.642] |
| variety-only rule | 0.609 [0.609, 0.609] | 0.645 [0.645, 0.645] | 0.609 [0.609, 0.609] | 0.645 [0.645, 0.645] |
| CNN, mean pool | 0.607 [0.585, 0.629] | 0.649 [0.627, 0.672] | pending | pending |
| CNN, attention pool | 0.586 [0.567, 0.606] | 0.652 [0.632, 0.673] | pending | pending |
| CNN + ECA, mean pool | 0.596 [0.577, 0.617] | 0.647 [0.627, 0.668] | pending | pending |
| CNN + ECA, attention pool | 0.598 [0.577, 0.618] | 0.639 [0.620, 0.660] | pending | pending |
| CNN + CBAM, mean pool | 0.616 [0.594, 0.638] | 0.664 [0.643, 0.687] | pending | pending |
| CNN + CBAM, attention pool | 0.601 [0.579, 0.624] | 0.657 [0.636, 0.679] | pending | pending |
| small ResNet, mean pool | 0.601 [0.578, 0.624] | 0.660 [0.638, 0.683] | pending | pending |
| small ResNet, attention pool | 0.591 [0.569, 0.612] | 0.651 [0.631, 0.671] | pending | pending |
| small ResNet + ECA, mean pool | 0.597 [0.575, 0.619] | 0.653 [0.631, 0.676] | pending | pending |
| small ResNet + ECA, attention pool | 0.581 [0.559, 0.604] | 0.645 [0.624, 0.666] | pending | pending |
| small ResNet + CBAM, mean pool | 0.592 [0.571, 0.614] | 0.649 [0.626, 0.671] | pending | pending |
| small ResNet + CBAM, attention pool | 0.612 [0.589, 0.635] | 0.663 [0.643, 0.683] | pending | pending |

Comparisons (# column): **1** against logistic regression, same data; **1b** against the variety-only rule (pooled sets); **2a** ECA / CBAM against the same network and pooling without them; **2b** attention against mean pooling, same network; **3** small ResNet against CNN, same module and pooling; **4** 204 bands against RGB, same model.

| # | A − B | bands | bal. acc. A vs B | Δ [95% CI] | seeds A > B / A < B |  | AUC A vs B | Δ [95% CI] | seeds A > B / A < B |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CNN, mean pool − logistic regression | 204 bands | 0.607 vs 0.590 | +0.017 [-0.010, +0.042] | 3 / 2 |  | 0.649 vs 0.634 | +0.015 [-0.003, +0.033] | 5 / 0 |  |
| 1 | CNN, attention pool − logistic regression | 204 bands | 0.586 vs 0.590 | -0.004 [-0.031, +0.021] | 1 / 4 |  | 0.652 vs 0.634 | +0.018 [-0.002, +0.037] | 3 / 2 |  |
| 1 | CNN + ECA, mean pool − logistic regression | 204 bands | 0.596 vs 0.590 | +0.006 [-0.019, +0.030] | 1 / 4 |  | 0.647 vs 0.634 | +0.013 [-0.006, +0.032] | 3 / 2 |  |
| 1 | CNN + ECA, attention pool − logistic regression | 204 bands | 0.598 vs 0.590 | +0.008 [-0.018, +0.031] | 1 / 4 |  | 0.639 vs 0.634 | +0.005 [-0.012, +0.022] | 1 / 4 |  |
| 1 | CNN + CBAM, mean pool − logistic regression | 204 bands | 0.616 vs 0.590 | +0.025 [-0.002, +0.053] | 3 / 1 |  | 0.664 vs 0.634 | +0.030 [+0.009, +0.051] | 5 / 0 | ✓ |
| 1 | CNN + CBAM, attention pool − logistic regression | 204 bands | 0.601 vs 0.590 | +0.011 [-0.015, +0.037] | 1 / 4 |  | 0.657 vs 0.634 | +0.023 [+0.003, +0.043] | 3 / 2 |  |
| 1 | small ResNet, mean pool − logistic regression | 204 bands | 0.601 vs 0.590 | +0.010 [-0.016, +0.036] | 3 / 2 |  | 0.660 vs 0.634 | +0.026 [+0.008, +0.044] | 5 / 0 | ✓ |
| 1 | small ResNet, attention pool − logistic regression | 204 bands | 0.591 vs 0.590 | +0.001 [-0.027, +0.025] | 1 / 4 |  | 0.651 vs 0.634 | +0.017 [-0.002, +0.037] | 3 / 2 |  |
| 1 | small ResNet + ECA, mean pool − logistic regression | 204 bands | 0.597 vs 0.590 | +0.007 [-0.019, +0.032] | 3 / 2 |  | 0.653 vs 0.634 | +0.019 [-0.000, +0.038] | 3 / 2 |  |
| 1 | small ResNet + ECA, attention pool − logistic regression | 204 bands | 0.581 vs 0.590 | -0.009 [-0.037, +0.017] | 3 / 2 |  | 0.645 vs 0.634 | +0.011 [-0.008, +0.030] | 3 / 2 |  |
| 1 | small ResNet + CBAM, mean pool − logistic regression | 204 bands | 0.592 vs 0.590 | +0.002 [-0.024, +0.028] | 1 / 4 |  | 0.649 vs 0.634 | +0.015 [-0.005, +0.034] | 3 / 2 |  |
| 1 | small ResNet + CBAM, attention pool − logistic regression | 204 bands | 0.612 vs 0.590 | +0.022 [-0.006, +0.050] | 3 / 2 |  | 0.663 vs 0.634 | +0.029 [+0.009, +0.049] | 4 / 1 | ✓ |
| 1b | logistic regression − variety-only rule | 204 bands | 0.590 vs 0.609 | -0.019 [-0.043, +0.006] | single fits |  | 0.634 vs 0.645 | -0.012 [-0.034, +0.012] | single fits |  |
| 1b | CNN, mean pool − variety-only rule | 204 bands | 0.607 vs 0.609 | -0.002 [-0.024, +0.020] | 0 / 5 |  | 0.649 vs 0.645 | +0.003 [-0.019, +0.027] | 0 / 5 |  |
| 1b | CNN, attention pool − variety-only rule | 204 bands | 0.586 vs 0.609 | -0.023 [-0.042, -0.003] | 0 / 5 | ✗ | 0.652 vs 0.645 | +0.007 [-0.013, +0.028] | 1 / 4 |  |
| 1b | CNN + ECA, mean pool − variety-only rule | 204 bands | 0.596 vs 0.609 | -0.013 [-0.032, +0.008] | 0 / 5 |  | 0.647 vs 0.645 | +0.001 [-0.019, +0.023] | 1 / 4 |  |
| 1b | CNN + ECA, attention pool − variety-only rule | 204 bands | 0.598 vs 0.609 | -0.011 [-0.032, +0.009] | 0 / 5 |  | 0.639 vs 0.645 | -0.006 [-0.026, +0.015] | 0 / 5 |  |
| 1b | CNN + CBAM, mean pool − variety-only rule | 204 bands | 0.616 vs 0.609 | +0.006 [-0.015, +0.029] | 0 / 5 |  | 0.664 vs 0.645 | +0.019 [-0.003, +0.041] | 4 / 1 |  |
| 1b | CNN + CBAM, attention pool − variety-only rule | 204 bands | 0.601 vs 0.609 | -0.008 [-0.030, +0.015] | 0 / 5 |  | 0.657 vs 0.645 | +0.011 [-0.010, +0.034] | 2 / 3 |  |
| 1b | small ResNet, mean pool − variety-only rule | 204 bands | 0.601 vs 0.609 | -0.009 [-0.031, +0.015] | 1 / 4 |  | 0.660 vs 0.645 | +0.015 [-0.007, +0.037] | 2 / 3 |  |
| 1b | small ResNet, attention pool − variety-only rule | 204 bands | 0.591 vs 0.609 | -0.018 [-0.040, +0.003] | 0 / 5 |  | 0.651 vs 0.645 | +0.006 [-0.014, +0.026] | 1 / 4 |  |
| 1b | small ResNet + ECA, mean pool − variety-only rule | 204 bands | 0.597 vs 0.609 | -0.012 [-0.034, +0.010] | 0 / 5 |  | 0.653 vs 0.645 | +0.008 [-0.015, +0.031] | 2 / 3 |  |
| 1b | small ResNet + ECA, attention pool − variety-only rule | 204 bands | 0.581 vs 0.609 | -0.028 [-0.050, -0.005] | 0 / 5 | ✗ | 0.645 vs 0.645 | -0.000 [-0.021, +0.021] | 1 / 4 |  |
| 1b | small ResNet + CBAM, mean pool − variety-only rule | 204 bands | 0.592 vs 0.609 | -0.017 [-0.038, +0.005] | 0 / 5 |  | 0.649 vs 0.645 | +0.004 [-0.019, +0.025] | 0 / 5 |  |
| 1b | small ResNet + CBAM, attention pool − variety-only rule | 204 bands | 0.612 vs 0.609 | +0.003 [-0.020, +0.026] | 2 / 3 |  | 0.663 vs 0.645 | +0.018 [-0.002, +0.038] | 3 / 2 |  |
| 1b | logistic regression − variety-only rule | RGB | 0.585 vs 0.609 | -0.024 [-0.050, +0.002] | single fits |  | 0.615 vs 0.645 | -0.031 [-0.058, -0.004] | single fits | ✗ |
| 2a | CNN + ECA, mean pool − CNN, mean pool | 204 bands | 0.596 vs 0.607 | -0.010 [-0.026, +0.006] | 1 / 4 |  | 0.647 vs 0.649 | -0.002 [-0.012, +0.008] | 2 / 3 |  |
| 2a | CNN + ECA, attention pool − CNN, attention pool | 204 bands | 0.598 vs 0.586 | +0.012 [-0.007, +0.030] | 1 / 4 |  | 0.639 vs 0.652 | -0.013 [-0.024, -0.003] | 1 / 4 | ✗ |
| 2a | CNN + CBAM, mean pool − CNN, mean pool | 204 bands | 0.616 vs 0.607 | +0.009 [-0.010, +0.030] | 2 / 3 |  | 0.664 vs 0.649 | +0.015 [+0.002, +0.028] | 5 / 0 | ✓ |
| 2a | CNN + CBAM, attention pool − CNN, attention pool | 204 bands | 0.601 vs 0.586 | +0.015 [-0.004, +0.035] | 3 / 2 |  | 0.657 vs 0.652 | +0.005 [-0.010, +0.019] | 3 / 2 |  |
| 2a | small ResNet + ECA, mean pool − small ResNet, mean pool | 204 bands | 0.597 vs 0.601 | -0.003 [-0.021, +0.014] | 2 / 3 |  | 0.653 vs 0.660 | -0.007 [-0.017, +0.003] | 2 / 3 |  |
| 2a | small ResNet + ECA, attention pool − small ResNet, attention pool | 204 bands | 0.581 vs 0.591 | -0.010 [-0.031, +0.010] | 4 / 1 |  | 0.645 vs 0.651 | -0.006 [-0.017, +0.005] | 3 / 2 |  |
| 2a | small ResNet + CBAM, mean pool − small ResNet, mean pool | 204 bands | 0.592 vs 0.601 | -0.008 [-0.026, +0.009] | 0 / 5 |  | 0.649 vs 0.660 | -0.011 [-0.022, -0.001] | 0 / 5 | ✗ |
| 2a | small ResNet + CBAM, attention pool − small ResNet, attention pool | 204 bands | 0.612 vs 0.591 | +0.021 [+0.001, +0.042] | 5 / 0 | ✓ | 0.663 vs 0.651 | +0.012 [+0.001, +0.022] | 4 / 1 | ✓ |
| 2b | CNN, attention pool − CNN, mean pool | 204 bands | 0.586 vs 0.607 | -0.020 [-0.036, -0.004] | 1 / 4 | ✗ | 0.652 vs 0.649 | +0.003 [-0.007, +0.014] | 3 / 2 |  |
| 2b | CNN + ECA, attention pool − CNN + ECA, mean pool | 204 bands | 0.598 vs 0.596 | +0.002 [-0.015, +0.018] | 1 / 4 |  | 0.639 vs 0.647 | -0.008 [-0.018, +0.002] | 2 / 3 |  |
| 2b | CNN + CBAM, attention pool − CNN + CBAM, mean pool | 204 bands | 0.601 vs 0.616 | -0.014 [-0.035, +0.006] | 2 / 3 |  | 0.657 vs 0.664 | -0.008 [-0.020, +0.005] | 1 / 4 |  |
| 2b | small ResNet, attention pool − small ResNet, mean pool | 204 bands | 0.591 vs 0.601 | -0.010 [-0.028, +0.011] | 1 / 4 |  | 0.651 vs 0.660 | -0.009 [-0.021, +0.003] | 1 / 4 |  |
| 2b | small ResNet + ECA, attention pool − small ResNet + ECA, mean pool | 204 bands | 0.581 vs 0.597 | -0.017 [-0.037, +0.004] | 1 / 4 |  | 0.645 vs 0.653 | -0.008 [-0.019, +0.004] | 3 / 2 |  |
| 2b | small ResNet + CBAM, attention pool − small ResNet + CBAM, mean pool | 204 bands | 0.612 vs 0.592 | +0.020 [-0.001, +0.041] | 4 / 1 |  | 0.663 vs 0.649 | +0.014 [+0.005, +0.024] | 4 / 1 | ✓ |
| 3 | small ResNet, mean pool − CNN, mean pool | 204 bands | 0.601 vs 0.607 | -0.006 [-0.022, +0.010] | 4 / 1 |  | 0.660 vs 0.649 | +0.011 [+0.003, +0.019] | 3 / 2 |  |
| 3 | small ResNet, attention pool − CNN, attention pool | 204 bands | 0.591 vs 0.586 | +0.004 [-0.013, +0.021] | 2 / 3 |  | 0.651 vs 0.652 | -0.001 [-0.012, +0.009] | 2 / 3 |  |
| 3 | small ResNet + ECA, mean pool − CNN + ECA, mean pool | 204 bands | 0.597 vs 0.596 | +0.001 [-0.016, +0.019] | 5 / 0 |  | 0.653 vs 0.647 | +0.006 [-0.005, +0.017] | 5 / 0 |  |
| 3 | small ResNet + ECA, attention pool − CNN + ECA, attention pool | 204 bands | 0.581 vs 0.598 | -0.017 [-0.038, +0.004] | 3 / 2 |  | 0.645 vs 0.639 | +0.006 [-0.005, +0.017] | 4 / 1 |  |
| 3 | small ResNet + CBAM, mean pool − CNN + CBAM, mean pool | 204 bands | 0.592 vs 0.616 | -0.023 [-0.047, -0.002] | 0 / 5 | ✗ | 0.649 vs 0.664 | -0.015 [-0.029, -0.002] | 0 / 5 | ✗ |
| 3 | small ResNet + CBAM, attention pool − CNN + CBAM, attention pool | 204 bands | 0.612 vs 0.601 | +0.011 [-0.012, +0.033] | 3 / 2 |  | 0.663 vs 0.657 | +0.006 [-0.006, +0.020] | 3 / 2 |  |
| 4 | logistic regression − logistic regression | 204 bands − RGB | 0.590 vs 0.585 | +0.005 [-0.025, +0.036] | single fits |  | 0.634 vs 0.615 | +0.019 [-0.008, +0.048] | single fits |  |

Marked, of 47 comparisons: balanced accuracy 1 ✓ and 4 ✗; AUC 6 ✓ and 4 ✗. Chance alone would put about 2.4 intervals per score off 0 (fewer once the seeds must agree).

## MOO (six heads)

One network with six outputs (4 varieties, red blotch, leafroll; Cab Franc masked for leafroll), small CNN or small ResNet with attention pooling, AdamW or SGD, 8 ways to combine the six gradients. Each task is scored at its own best validation epoch; the single-task CNNs, logistic regression and the variety-only rule are the same rows as above, on the same resamples.

SGD learning rate chosen per method on 204 bands (seed 0, every fold, lowest mean validation loss of the six heads); RGB reuses it:

| method | six-head CNN | six-head small ResNet |
|---|---|---|
| LS | pending | pending |
| MGDA | pending | pending |
| MGDA-GN | pending | pending |
| MGDA-LN | pending | pending |
| MGDA-LGN | pending | pending |
| SVD-PGA | pending | pending |
| OSNMF-PGA | pending | pending |
| UPGrad | pending | pending |

### Variety (4 classes), all 998 leaves

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.885 [0.870, 0.900] | 0.985 [0.981, 0.988] | 0.675 [0.649, 0.702] | 0.864 [0.848, 0.880] |
| CNN, attention pool | 0.885 [0.873, 0.897] | 0.986 [0.983, 0.988] | pending | pending |
| small ResNet, attention pool | 0.895 [0.885, 0.905] | 0.984 [0.981, 0.987] | pending | pending |
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |

### Leafroll, Baco + Chardonnay + Vidal (Cab Franc masked)

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.587 [0.557, 0.616] | 0.619 [0.594, 0.644] | 0.573 [0.546, 0.599] | 0.622 [0.595, 0.647] |
| variety-only rule | 0.676 [0.676, 0.676] | 0.652 [0.652, 0.652] | 0.676 [0.676, 0.676] | 0.652 [0.652, 0.652] |
| CNN, attention pool | 0.595 [0.576, 0.616] | 0.624 [0.603, 0.644] | pending | pending |
| small ResNet, attention pool | 0.611 [0.590, 0.635] | 0.661 [0.640, 0.682] | pending | pending |
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |

### Red blotch, all 998 leaves

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.663 [0.643, 0.684] | 0.724 [0.705, 0.742] | 0.641 [0.619, 0.664] | 0.683 [0.660, 0.706] |
| variety-only rule | 0.706 [0.706, 0.706] | 0.764 [0.764, 0.764] | 0.706 [0.706, 0.706] | 0.764 [0.764, 0.764] |
| CNN, attention pool | 0.677 [0.660, 0.694] | 0.748 [0.732, 0.763] | pending | pending |
| small ResNet, attention pool | 0.686 [0.667, 0.704] | 0.752 [0.737, 0.767] | pending | pending |
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |

### Red blotch, all 998 leaves, scored without Cab Franc

| model | 204 bands: bal. acc. [95% CI] | 204 bands: AUC [95% CI] | RGB: bal. acc. [95% CI] | RGB: AUC [95% CI] |
|---|---|---|---|---|
| logistic regression | 0.590 [0.566, 0.615] | 0.634 [0.611, 0.658] | 0.585 [0.559, 0.611] | 0.615 [0.587, 0.642] |
| variety-only rule | 0.609 [0.609, 0.609] | 0.645 [0.645, 0.645] | 0.609 [0.609, 0.609] | 0.645 [0.645, 0.645] |
| CNN, attention pool | 0.586 [0.567, 0.606] | 0.652 [0.632, 0.673] | pending | pending |
| small ResNet, attention pool | 0.591 [0.569, 0.612] | 0.651 [0.631, 0.671] | pending | pending |
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |

### One checkpoint for all six heads

Balanced accuracy · AUC when one epoch (the lowest summed validation loss of all six heads) serves every task. No comparisons; shown beside the per-task checkpoints above.

204 bands:

| model | Variety (4 classes), all 998 leaves | Leafroll, Baco + Chardonnay + Vidal (Cab Franc masked) | Red blotch, all 998 leaves | Red blotch, all 998 leaves, scored without Cab Franc |
|---|---|---|---|---|
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |

RGB:

| model | Variety (4 classes), all 998 leaves | Leafroll, Baco + Chardonnay + Vidal (Cab Franc masked) | Red blotch, all 998 leaves | Red blotch, all 998 leaves, scored without Cab Franc |
|---|---|---|---|---|
| six-head CNN, AdamW, LS | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, AdamW, UPGrad | pending | pending | pending | pending |
| six-head CNN, SGD, LS | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head CNN, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head CNN, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head CNN, SGD, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, AdamW, LS | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, AdamW, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, AdamW, UPGrad | pending | pending | pending | pending |
| six-head small ResNet, SGD, LS | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-GN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LN | pending | pending | pending | pending |
| six-head small ResNet, SGD, MGDA-LGN | pending | pending | pending | pending |
| six-head small ResNet, SGD, SVD-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, OSNMF-PGA | pending | pending | pending | pending |
| six-head small ResNet, SGD, UPGrad | pending | pending | pending | pending |
