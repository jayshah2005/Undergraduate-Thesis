# Six-head CNN: LS, MGDA and PGA against the single-task models

One CNN with six sigmoid heads (Baco, Cab Franc, Chardonnay, Vidal, red blotch, leafroll), 998 leaves, 204 bands. The methods differ only in how the six heads' gradients become one update:

- **LS**: their sum (one backward pass on the summed loss).
- **MGDA** (Sener & Koltun 2018): the shortest mix of the six gradients, over all weights. **-GN / -LN / -LGN** first divide each gradient by its norm, its loss, or both (the variants in LibMTL, Lin & Zhang 2023).
- **SVD-PGA / OSNMF-PGA**: the Principal Gradient Alignment aggregators of the local torchjd (commit 04d1bef, not yet published), default settings.

Same row folds (test Rk, validation R(k+1), train on the other 3), seeds and 3-seed ensemble as the variety and disease CNNs, scored with their own report code. Rule, set before the runs: a difference counts only where its paired 95% interval excludes 0 (marked *).

## Summary

Balanced accuracy (chance: variety 0.25, disease 0.5). Leafroll never includes Cab Franc (it has too few leafroll leaves): 748 leaves. Red blotch is shown on all 998 leaves and without Cab Franc (748 leaves): 241 of Cab Franc's 250 leaves have red blotch, so a model gets them right from the variety alone. Both red-blotch columns score the same trained models.

| model | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| logistic regression | 0.874 | 0.572 | 0.647 | 0.594 |
| CNN, mean pool (single-task) | 0.898 | 0.586 | 0.691 | 0.604 |
| CNN, attention pool (single-task) | 0.895 | 0.597 | 0.666 | 0.580 |
| small ResNet, mean pool (single-task) | 0.907 | 0.585 | 0.682 | 0.596 |
| small ResNet, attention pool (single-task) | 0.902 | 0.612 | 0.686 | 0.594 |
| six-head, LS | 0.833 | 0.492 | 0.660 | 0.625 |
| six-head, MGDA | 0.825 | 0.495 | 0.598 | 0.588 |
| six-head, MGDA-GN | 0.819 | 0.532 | 0.632 | 0.615 |
| six-head, MGDA-LN | 0.833 | 0.499 | 0.622 | 0.595 |
| six-head, MGDA-LGN | 0.806 | 0.504 | 0.642 | 0.613 |
| six-head, SVD-PGA | 0.759 | 0.531 | 0.610 | 0.592 |
| six-head, OSNMF-PGA | 0.840 | 0.488 | 0.633 | 0.610 |

## Each method against LS

Paired difference in balanced accuracy [95% CI], same resampled leaves; * = interval excludes 0.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| MGDA − LS | -0.008 [-0.027, +0.012] | +0.003 [-0.017, +0.024] | -0.062 [-0.079, -0.046] * | -0.038 [-0.056, -0.019] * |
| MGDA-GN − LS | -0.014 [-0.034, +0.006] | +0.040 [+0.022, +0.058] * | -0.028 [-0.044, -0.011] * | -0.011 [-0.029, +0.009] |
| MGDA-LN − LS | +0.000 [-0.020, +0.020] | +0.008 [-0.016, +0.031] | -0.038 [-0.056, -0.022] * | -0.031 [-0.050, -0.012] * |
| MGDA-LGN − LS | -0.027 [-0.046, -0.008] * | +0.012 [-0.009, +0.034] | -0.018 [-0.035, -0.002] * | -0.013 [-0.031, +0.005] |
| SVD-PGA − LS | -0.074 [-0.099, -0.049] * | +0.039 [+0.016, +0.061] * | -0.051 [-0.070, -0.030] * | -0.033 [-0.054, -0.010] * |
| OSNMF-PGA − LS | +0.007 [-0.014, +0.026] | -0.004 [-0.023, +0.015] | -0.027 [-0.043, -0.011] * | -0.016 [-0.033, +0.003] |

## Six-head against the single-task CNN, attention pool

The CNN, attention pool is the closest single-task model (same pooling). Paired, as above.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| LS − CNN, attention pool | -0.062 [-0.087, -0.038] * | -0.106 [-0.132, -0.079] * | -0.006 [-0.029, +0.016] | +0.046 [+0.020, +0.071] * |
| MGDA − CNN, attention pool | -0.070 [-0.095, -0.045] * | -0.102 [-0.125, -0.080] * | -0.068 [-0.089, -0.048] * | +0.008 [-0.017, +0.033] |
| MGDA-GN − CNN, attention pool | -0.076 [-0.102, -0.051] * | -0.066 [-0.090, -0.040] * | -0.034 [-0.057, -0.011] * | +0.035 [+0.010, +0.062] * |
| MGDA-LN − CNN, attention pool | -0.062 [-0.088, -0.035] * | -0.098 [-0.128, -0.069] * | -0.044 [-0.070, -0.018] * | +0.015 [-0.014, +0.045] |
| MGDA-LGN − CNN, attention pool | -0.089 [-0.114, -0.065] * | -0.094 [-0.117, -0.069] * | -0.024 [-0.049, -0.001] * | +0.033 [+0.004, +0.061] * |
| SVD-PGA − CNN, attention pool | -0.136 [-0.162, -0.110] * | -0.067 [-0.094, -0.040] * | -0.057 [-0.083, -0.029] * | +0.013 [-0.019, +0.045] |
| OSNMF-PGA − CNN, attention pool | -0.055 [-0.081, -0.029] * | -0.109 [-0.133, -0.087] * | -0.033 [-0.056, -0.011] * | +0.030 [+0.005, +0.057] * |

## Weight per head

Each epoch's mean weights divided by the sum of their absolute values, averaged over every epoch of a method's 15 trainings (range over epochs). For MGDA these are α; a PGA weight can be negative.

| method | Baco | Cab Franc | Chardonnay | Vidal | red blotch | leafroll |
|---|---|---|---|---|---|---|
| MGDA | 0.39 (0.04–0.91) | 0.11 (0.01–0.29) | 0.17 (0.01–0.54) | 0.09 (0.00–0.31) | 0.12 (0.00–0.31) | 0.13 (0.01–0.29) |
| MGDA-GN | 0.31 (0.13–0.77) | 0.14 (0.04–0.21) | 0.15 (0.04–0.23) | 0.13 (0.03–0.22) | 0.14 (0.04–0.21) | 0.14 (0.04–0.22) |
| MGDA-LN | 0.18 (0.01–0.72) | 0.11 (0.01–0.29) | 0.08 (0.00–0.49) | 0.13 (0.01–0.33) | 0.27 (0.03–0.50) | 0.23 (0.03–0.50) |
| MGDA-LGN | 0.17 (0.01–0.73) | 0.16 (0.00–0.57) | 0.19 (0.00–0.76) | 0.15 (0.00–0.50) | 0.18 (0.00–0.38) | 0.15 (0.00–0.34) |
| SVD-PGA | 0.54 (0.00–1.00) | 0.09 (-0.00–0.68) | 0.23 (-0.01–0.99) | 0.05 (0.00–0.27) | 0.05 (0.00–0.30) | 0.05 (0.00–0.29) |
| OSNMF-PGA | 0.56 (-0.59–1.00) | 0.06 (-0.00–0.29) | 0.25 (-0.02–0.99) | 0.05 (-0.01–0.34) | 0.04 (0.00–0.30) | 0.04 (0.00–0.28) |

## Details

### Variety, per test row

| model | bal. acc. [95% CI] | Δ vs logistic regression | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| CNN, mean pool | 0.898 [0.880, 0.916] | +0.024 [+0.007, +0.041] * | 0.595 | 0.975 | 0.975 | 0.975 | 0.970 |
| CNN, attention pool | 0.895 [0.876, 0.914] | +0.021 [+0.005, +0.038] * | 0.620 | 0.970 | 0.995 | 0.980 | 0.910 |
| small ResNet, mean pool | 0.907 [0.890, 0.924] | +0.033 [+0.018, +0.049] * | 0.610 | 0.960 | 0.995 | 0.990 | 0.980 |
| small ResNet, attention pool | 0.902 [0.884, 0.920] | +0.028 [+0.013, +0.044] * | 0.595 | 0.985 | 1.000 | 0.965 | 0.965 |
| six-head, LS | 0.833 [0.810, 0.854] | -0.041 [-0.067, -0.016] * | 0.835 | 0.870 | 0.920 | 0.870 | 0.670 |
| six-head, MGDA | 0.825 [0.801, 0.847] | -0.049 [-0.074, -0.025] * | 0.830 | 0.885 | 0.880 | 0.855 | 0.675 |
| six-head, MGDA-GN | 0.819 [0.795, 0.841] | -0.055 [-0.081, -0.029] * | 0.860 | 0.795 | 0.910 | 0.845 | 0.685 |
| six-head, MGDA-LN | 0.833 [0.811, 0.855] | -0.041 [-0.067, -0.014] * | 0.865 | 0.935 | 0.845 | 0.835 | 0.685 |
| six-head, MGDA-LGN | 0.806 [0.782, 0.828] | -0.068 [-0.094, -0.043] * | 0.775 | 0.895 | 0.910 | 0.780 | 0.670 |
| six-head, SVD-PGA | 0.759 [0.734, 0.783] | -0.115 [-0.140, -0.089] * | 0.715 | 0.865 | 0.915 | 0.700 | 0.600 |
| six-head, OSNMF-PGA | 0.840 [0.817, 0.862] | -0.034 [-0.060, -0.008] * | 0.865 | 0.900 | 0.945 | 0.875 | 0.615 |

### Leafroll (748 leaves, 255 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.471 | 0.478 | 0.522 |
| CNN, mean pool | 0.586 [0.566, 0.606] | +0.014 [-0.021, +0.048] | 0.494 | 0.457 | 0.518 |
| CNN, attention pool | 0.597 [0.578, 0.618] | +0.026 [-0.010, +0.061] | 0.516 | 0.454 | 0.468 |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | +0.014 [-0.022, +0.049] | 0.503 | 0.442 | 0.505 |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | +0.040 [+0.002, +0.075] * | 0.518 | 0.520 | 0.502 |
| six-head, LS | 0.492 [0.473, 0.511] | -0.080 [-0.116, -0.046] * | 0.412 | 0.425 | 0.500 |
| six-head, MGDA | 0.495 [0.484, 0.506] | -0.076 [-0.110, -0.043] * | 0.463 | 0.496 | 0.500 |
| six-head, MGDA-GN | 0.532 [0.516, 0.549] | -0.040 [-0.074, -0.006] * | 0.514 | 0.500 | 0.500 |
| six-head, MGDA-LN | 0.499 [0.478, 0.521] | -0.072 [-0.110, -0.036] * | 0.422 | 0.437 | 0.500 |
| six-head, MGDA-LGN | 0.504 [0.490, 0.519] | -0.068 [-0.101, -0.034] * | 0.473 | 0.501 | 0.500 |
| six-head, SVD-PGA | 0.531 [0.513, 0.550] | -0.041 [-0.077, -0.005] * | 0.507 | 0.491 | 0.500 |
| six-head, OSNMF-PGA | 0.488 [0.477, 0.499] | -0.083 [-0.117, -0.051] * | 0.494 | 0.431 | 0.500 |

### Red blotch, all leaves (998 leaves, 637 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.456 | 0.475 | 0.540 | 0.522 |
| CNN, mean pool | 0.691 [0.672, 0.709] | +0.044 [+0.019, +0.070] * | 0.499 | 0.522 | 0.555 | 0.539 |
| CNN, attention pool | 0.666 [0.646, 0.685] | +0.020 [-0.007, +0.047] | 0.500 | 0.510 | 0.539 | 0.505 |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | +0.036 [+0.008, +0.063] * | 0.504 | 0.512 | 0.555 | 0.537 |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | +0.040 [+0.014, +0.065] * | 0.474 | 0.524 | 0.538 | 0.541 |
| six-head, LS | 0.660 [0.646, 0.674] | +0.013 [-0.012, +0.038] | 0.527 | 0.533 | 0.539 | 0.507 |
| six-head, MGDA | 0.598 [0.584, 0.610] | -0.049 [-0.071, -0.023] * | 0.500 | 0.545 | 0.593 | 0.500 |
| six-head, MGDA-GN | 0.632 [0.615, 0.647] | -0.014 [-0.040, +0.010] | 0.483 | 0.535 | 0.606 | 0.500 |
| six-head, MGDA-LN | 0.622 [0.604, 0.640] | -0.025 [-0.051, +0.000] | 0.475 | 0.539 | 0.537 | 0.506 |
| six-head, MGDA-LGN | 0.642 [0.623, 0.658] | -0.005 [-0.032, +0.023] | 0.512 | 0.475 | 0.534 | 0.510 |
| six-head, SVD-PGA | 0.610 [0.589, 0.630] | -0.037 [-0.063, -0.010] * | 0.469 | 0.508 | 0.591 | 0.508 |
| six-head, OSNMF-PGA | 0.633 [0.618, 0.647] | -0.014 [-0.038, +0.010] | 0.493 | 0.494 | 0.564 | 0.500 |

### Red blotch without Cab Franc (748 leaves, 396 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.594 [0.569, 0.618] | — | 0.456 | 0.540 | 0.522 |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.010 [-0.019, +0.040] | 0.499 | 0.555 | 0.539 |
| CNN, attention pool | 0.580 [0.556, 0.602] | -0.014 [-0.047, +0.017] | 0.500 | 0.539 | 0.505 |
| small ResNet, mean pool | 0.596 [0.572, 0.620] | +0.003 [-0.029, +0.034] | 0.504 | 0.555 | 0.537 |
| small ResNet, attention pool | 0.594 [0.572, 0.615] | +0.000 [-0.030, +0.031] | 0.474 | 0.538 | 0.541 |
| six-head, LS | 0.625 [0.610, 0.640] | +0.032 [+0.004, +0.059] * | 0.527 | 0.539 | 0.507 |
| six-head, MGDA | 0.588 [0.573, 0.603] | -0.006 [-0.032, +0.022] | 0.500 | 0.593 | 0.500 |
| six-head, MGDA-GN | 0.615 [0.596, 0.632] | +0.021 [-0.008, +0.049] | 0.483 | 0.606 | 0.500 |
| six-head, MGDA-LN | 0.595 [0.575, 0.615] | +0.001 [-0.028, +0.030] | 0.475 | 0.537 | 0.506 |
| six-head, MGDA-LGN | 0.613 [0.593, 0.632] | +0.019 [-0.012, +0.049] | 0.512 | 0.534 | 0.510 |
| six-head, SVD-PGA | 0.592 [0.572, 0.614] | -0.001 [-0.033, +0.029] | 0.469 | 0.591 | 0.508 |
| six-head, OSNMF-PGA | 0.610 [0.594, 0.627] | +0.016 [-0.013, +0.045] | 0.493 | 0.564 | 0.500 |

## Notes

- Median best epoch: LS 25, MGDA 20, MGDA-GN 8, MGDA-LN 13, MGDA-LGN 9, SVD-PGA 8, OSNMF-PGA 8. Trainings that reached the 40-epoch limit (early stopping never triggered): LS 6 of 15, MGDA 3 of 15, MGDA-GN 0 of 15, MGDA-LN 0 of 15, MGDA-LGN 0 of 15, SVD-PGA 1 of 15, OSNMF-PGA 0 of 15.
- The single-task models in the red blotch columns trained on all 998 leaves. Trained on Baco, Chardonnay and Vidal only, the disease CNN scores 0.619 (mean pool) and 0.606 (attention pool) and logistic regression 0.600 (`disease_classification/REPORT.md`).
- The six-head CNN is not the same network as the single-task CNNs: two 3 × 3 convolutions at full resolution and no batch norm, no rotations or flips, a 211 px canvas, one checkpoint for all six heads (lowest summed validation loss), and its leafroll and red blotch heads also train on Cab Franc. So the clean comparison is between the six-head methods; against the single-task models the network differs too.
