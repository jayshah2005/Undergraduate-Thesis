# Six-head CNN in the disease CNN's setup: LS, MGDA and PGA against the single-task models

One CNN with six sigmoid heads (Baco, Cab Franc, Chardonnay, Vidal, red blotch, leafroll), 998 leaves, 204 bands. The methods differ only in how the six heads' gradients become one update:

- **LS**: their sum (one backward pass on the summed loss).
- **MGDA** (Sener & Koltun 2018): the shortest mix of the six gradients, over all weights. **-GN / -LN / -LGN** first divide each gradient by its norm, its loss, or both (the variants in LibMTL, Lin & Zhang 2023).
- **SVD-PGA / OSNMF-PGA**: the Principal Gradient Alignment aggregators of the local torchjd (commit 04d1bef, not yet published), default settings.
- **UPGrad** (Quinton & Rey 2024, "Jacobian Descent for Multi-Objective Optimization"): each gradient projected onto the dual cone of all six (so it conflicts with none), then averaged; torchjd's default.

The six-head CNN here has the disease CNN's setup (`run_moo_disease.py`): the small ResNet without max pooling, attention pooling, dropout 0.5, the 212 px canvas with SNV, rotations and flips, AdamW, up to 80 epochs with patience 15, class-weighted losses, leafroll without Cab Franc. Like a single-task model, each task keeps its own best validation epoch (variety: the 4 variety heads' summed loss); the rows marked '(one checkpoint)' keep one epoch for all six heads instead. What still differs from the single-task models: variety comes from 4 yes/no heads instead of one 4-class softmax, and one network serves all six objectives, which is what is being tested.

Same row folds (test Rk, validation R(k+1), train on the other 3), seeds and 3-seed ensemble as the variety and disease CNNs, scored with their own report code. Rule, set before the runs: a difference counts only where its paired 95% interval excludes 0 (marked *).

## Summary

Balanced accuracy (chance: variety 0.25, disease 0.5). Leafroll never includes Cab Franc (it has too few leafroll leaves): 748 leaves. Red blotch is shown on all 998 leaves and without Cab Franc (748 leaves): 241 of Cab Franc's 250 leaves have red blotch, so a model gets them right from the variety alone. Both red-blotch columns score the same trained models.

| model | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| logistic regression | 0.874 | 0.572 | 0.647 | 0.594 |
| small ResNet (no max pool), attention pool (single-task) | 0.905 | 0.602 | 0.673 | 0.589 |
| six-head, LS | 0.893 | 0.596 | 0.703 | 0.619 |
| six-head, MGDA | 0.842 | 0.600 | 0.693 | 0.605 |
| six-head, MGDA-GN | 0.883 | 0.599 | 0.683 | 0.581 |
| six-head, MGDA-LN | 0.897 | 0.634 | 0.671 | 0.562 |
| six-head, MGDA-LGN | 0.864 | 0.598 | 0.699 | 0.608 |
| six-head, UPGrad | 0.896 | 0.600 | 0.698 | 0.616 |

One checkpoint for all six heads (lowest summed validation loss), as one deployed model would be:

| model | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| six-head, LS (one checkpoint) | 0.893 | 0.647 | 0.685 | 0.587 |
| six-head, MGDA (one checkpoint) | 0.852 | 0.609 | 0.676 | 0.598 |
| six-head, MGDA-GN (one checkpoint) | 0.890 | 0.552 | 0.686 | 0.607 |
| six-head, MGDA-LN (one checkpoint) | 0.897 | 0.585 | 0.705 | 0.618 |
| six-head, MGDA-LGN (one checkpoint) | 0.861 | 0.606 | 0.691 | 0.605 |
| six-head, UPGrad (one checkpoint) | 0.896 | 0.640 | 0.699 | 0.602 |

## Each method against LS

Paired difference in balanced accuracy [95% CI], same resampled leaves; * = interval excludes 0.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| MGDA − LS | -0.051 [-0.071, -0.032] * | +0.004 [-0.015, +0.023] | -0.009 [-0.025, +0.007] | -0.014 [-0.033, +0.005] |
| MGDA-GN − LS | -0.010 [-0.029, +0.009] | +0.003 [-0.012, +0.018] | -0.020 [-0.036, -0.004] * | -0.038 [-0.058, -0.018] * |
| MGDA-LN − LS | +0.004 [-0.010, +0.018] | +0.038 [+0.022, +0.055] * | -0.032 [-0.045, -0.017] * | -0.057 [-0.073, -0.040] * |
| MGDA-LGN − LS | -0.029 [-0.050, -0.010] * | +0.001 [-0.018, +0.020] | -0.004 [-0.019, +0.011] | -0.011 [-0.030, +0.008] |
| UPGrad − LS | +0.003 [-0.009, +0.015] | +0.004 [-0.011, +0.020] | -0.005 [-0.017, +0.009] | -0.003 [-0.020, +0.013] |

## Six-head against the single-task small ResNet (no max pool), attention pool

The small ResNet (no max pool), attention pool is the closest single-task model (same pooling, backbone and training). Paired, as above.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| LS − small ResNet (no max pool), attention pool | -0.012 [-0.025, +0.001] | -0.006 [-0.028, +0.014] | +0.029 [+0.007, +0.051] * | +0.030 [+0.006, +0.054] * |
| MGDA − small ResNet (no max pool), attention pool | -0.063 [-0.082, -0.044] * | -0.002 [-0.022, +0.020] | +0.020 [+0.000, +0.040] * | +0.016 [-0.005, +0.038] |
| MGDA-GN − small ResNet (no max pool), attention pool | -0.022 [-0.038, -0.006] * | -0.003 [-0.023, +0.017] | +0.010 [-0.012, +0.031] | -0.008 [-0.034, +0.017] |
| MGDA-LN − small ResNet (no max pool), attention pool | -0.008 [-0.023, +0.005] | +0.032 [+0.011, +0.053] * | -0.002 [-0.023, +0.017] | -0.026 [-0.049, -0.003] * |
| MGDA-LGN − small ResNet (no max pool), attention pool | -0.041 [-0.059, -0.023] * | -0.005 [-0.025, +0.016] | +0.025 [+0.006, +0.046] * | +0.019 [-0.005, +0.042] |
| UPGrad − small ResNet (no max pool), attention pool | -0.009 [-0.021, +0.003] | -0.002 [-0.022, +0.017] | +0.025 [+0.003, +0.046] * | +0.027 [+0.001, +0.052] * |

## Weight per head

Each epoch's mean weights divided by the sum of their absolute values, averaged over every epoch of a method's 15 trainings (range over epochs). For MGDA these are α; a PGA weight can be negative.

| method | Baco | Cab Franc | Chardonnay | Vidal | red blotch | leafroll |
|---|---|---|---|---|---|---|
| MGDA | 0.53 (0.07–0.89) | 0.12 (0.00–0.75) | 0.28 (0.01–0.71) | 0.02 (0.00–0.14) | 0.05 (0.00–0.38) | 0.01 (0.00–0.11) |
| MGDA-GN | 0.24 (0.00–0.70) | 0.13 (0.00–0.70) | 0.40 (0.10–0.92) | 0.07 (0.00–0.19) | 0.08 (0.00–0.22) | 0.08 (0.00–0.21) |
| MGDA-LN | 0.20 (0.00–0.84) | 0.10 (0.00–0.65) | 0.21 (0.02–0.79) | 0.08 (0.00–0.29) | 0.24 (0.02–0.52) | 0.18 (0.02–0.39) |
| MGDA-LGN | 0.10 (0.00–0.60) | 0.14 (0.02–0.67) | 0.11 (0.01–0.66) | 0.15 (0.04–0.45) | 0.13 (0.00–0.29) | 0.36 (0.01–0.61) |
| UPGrad | 0.24 (0.14–0.42) | 0.17 (0.09–0.34) | 0.23 (0.13–0.38) | 0.14 (0.08–0.31) | 0.12 (0.08–0.21) | 0.11 (0.07–0.16) |

## Details

### Variety, per test row

| model | bal. acc. [95% CI] | Δ vs logistic regression | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| small ResNet (no max pool), attention pool | 0.905 [0.888, 0.923] | +0.031 [+0.015, +0.047] * | 0.620 | 0.960 | 1.000 | 0.985 | 0.960 |
| six-head, LS | 0.893 [0.875, 0.911] | +0.019 [+0.003, +0.037] * | 0.545 | 0.985 | 0.995 | 0.975 | 0.965 |
| six-head, LS (one checkpoint) | 0.893 [0.875, 0.911] | +0.019 [+0.003, +0.037] * | 0.545 | 0.985 | 0.995 | 0.975 | 0.965 |
| six-head, MGDA | 0.842 [0.821, 0.863] | -0.032 [-0.051, -0.014] * | 0.630 | 0.870 | 0.970 | 0.895 | 0.845 |
| six-head, MGDA (one checkpoint) | 0.852 [0.831, 0.872] | -0.022 [-0.040, -0.005] * | 0.605 | 0.895 | 0.970 | 0.895 | 0.895 |
| six-head, MGDA-GN | 0.883 [0.864, 0.901] | +0.009 [-0.010, +0.028] | 0.655 | 0.940 | 0.980 | 0.980 | 0.860 |
| six-head, MGDA-GN (one checkpoint) | 0.890 [0.871, 0.908] | +0.016 [-0.002, +0.034] | 0.655 | 0.970 | 0.985 | 0.980 | 0.860 |
| six-head, MGDA-LN | 0.897 [0.879, 0.915] | +0.023 [+0.005, +0.040] * | 0.630 | 0.985 | 0.995 | 0.920 | 0.955 |
| six-head, MGDA-LN (one checkpoint) | 0.897 [0.879, 0.915] | +0.023 [+0.005, +0.040] * | 0.630 | 0.985 | 0.995 | 0.920 | 0.955 |
| six-head, MGDA-LGN | 0.864 [0.842, 0.884] | -0.010 [-0.028, +0.008] | 0.685 | 0.930 | 0.975 | 0.885 | 0.845 |
| six-head, MGDA-LGN (one checkpoint) | 0.861 [0.840, 0.881] | -0.013 [-0.031, +0.006] | 0.685 | 0.920 | 0.975 | 0.885 | 0.840 |
| six-head, UPGrad | 0.896 [0.878, 0.915] | +0.022 [+0.008, +0.037] * | 0.610 | 0.975 | 1.000 | 0.970 | 0.925 |
| six-head, UPGrad (one checkpoint) | 0.896 [0.878, 0.915] | +0.022 [+0.008, +0.037] * | 0.610 | 0.975 | 1.000 | 0.970 | 0.925 |

### Leafroll (748 leaves, 255 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.471 | 0.478 | 0.522 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.524 | 0.475 | 0.496 |
| six-head, LS | 0.596 [0.581, 0.611] | +0.024 [-0.010, +0.057] | 0.536 | 0.424 | 0.495 |
| six-head, LS (one checkpoint) | 0.647 [0.635, 0.658] | +0.075 [+0.041, +0.108] * | 0.505 | 0.500 | 0.499 |
| six-head, MGDA | 0.600 [0.584, 0.617] | +0.029 [-0.006, +0.063] | 0.500 | 0.481 | 0.485 |
| six-head, MGDA (one checkpoint) | 0.609 [0.590, 0.627] | +0.037 [-0.000, +0.074] | 0.469 | 0.463 | 0.512 |
| six-head, MGDA-GN | 0.599 [0.587, 0.611] | +0.027 [-0.007, +0.060] | 0.507 | 0.439 | 0.504 |
| six-head, MGDA-GN (one checkpoint) | 0.552 [0.544, 0.561] | -0.019 [-0.053, +0.015] | 0.446 | 0.427 | 0.541 |
| six-head, MGDA-LN | 0.634 [0.620, 0.648] | +0.062 [+0.028, +0.095] * | 0.588 | 0.446 | 0.493 |
| six-head, MGDA-LN (one checkpoint) | 0.585 [0.572, 0.598] | +0.013 [-0.023, +0.047] | 0.465 | 0.479 | 0.495 |
| six-head, MGDA-LGN | 0.598 [0.583, 0.611] | +0.026 [-0.009, +0.059] | 0.511 | 0.465 | 0.550 |
| six-head, MGDA-LGN (one checkpoint) | 0.606 [0.591, 0.621] | +0.035 [-0.002, +0.070] | 0.470 | 0.479 | 0.541 |
| six-head, UPGrad | 0.600 [0.586, 0.616] | +0.029 [-0.006, +0.061] | 0.498 | 0.444 | 0.493 |
| six-head, UPGrad (one checkpoint) | 0.640 [0.625, 0.653] | +0.068 [+0.033, +0.102] * | 0.517 | 0.483 | 0.509 |

### Red blotch, all leaves (998 leaves, 637 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | +0.027 [-0.000, +0.053] | 0.466 | 0.563 | 0.527 | 0.530 |
| six-head, LS | 0.703 [0.688, 0.717] | +0.056 [+0.031, +0.082] * | 0.517 | 0.498 | 0.529 | 0.535 |
| six-head, LS (one checkpoint) | 0.685 [0.674, 0.697] | +0.039 [+0.015, +0.063] * | 0.482 | 0.475 | 0.500 | 0.492 |
| six-head, MGDA | 0.693 [0.680, 0.707] | +0.047 [+0.023, +0.071] * | 0.475 | 0.539 | 0.522 | 0.529 |
| six-head, MGDA (one checkpoint) | 0.676 [0.665, 0.687] | +0.030 [+0.007, +0.054] * | 0.474 | 0.557 | 0.508 | 0.495 |
| six-head, MGDA-GN | 0.683 [0.669, 0.698] | +0.037 [+0.012, +0.063] * | 0.520 | 0.498 | 0.524 | 0.446 |
| six-head, MGDA-GN (one checkpoint) | 0.686 [0.675, 0.697] | +0.039 [+0.017, +0.063] * | 0.488 | 0.553 | 0.525 | 0.496 |
| six-head, MGDA-LN | 0.671 [0.658, 0.683] | +0.025 [+0.001, +0.049] * | 0.493 | 0.494 | 0.499 | 0.452 |
| six-head, MGDA-LN (one checkpoint) | 0.705 [0.695, 0.715] | +0.058 [+0.036, +0.081] * | 0.508 | 0.492 | 0.511 | 0.513 |
| six-head, MGDA-LGN | 0.699 [0.685, 0.712] | +0.052 [+0.028, +0.075] * | 0.482 | 0.494 | 0.527 | 0.552 |
| six-head, MGDA-LGN (one checkpoint) | 0.691 [0.679, 0.705] | +0.045 [+0.022, +0.068] * | 0.484 | 0.547 | 0.528 | 0.490 |
| six-head, UPGrad | 0.698 [0.684, 0.712] | +0.052 [+0.026, +0.078] * | 0.512 | 0.498 | 0.519 | 0.511 |
| six-head, UPGrad (one checkpoint) | 0.699 [0.693, 0.706] | +0.053 [+0.031, +0.074] * | 0.486 | 0.498 | 0.500 | 0.498 |

### Red blotch without Cab Franc (748 leaves, 396 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.594 [0.569, 0.618] | — | 0.456 | 0.540 | 0.522 |
| small ResNet (no max pool), attention pool | 0.589 [0.567, 0.610] | -0.005 [-0.036, +0.025] | 0.466 | 0.527 | 0.530 |
| six-head, LS | 0.619 [0.601, 0.636] | +0.025 [-0.004, +0.054] | 0.517 | 0.529 | 0.535 |
| six-head, LS (one checkpoint) | 0.587 [0.575, 0.599] | -0.007 [-0.033, +0.019] | 0.482 | 0.500 | 0.492 |
| six-head, MGDA | 0.605 [0.589, 0.620] | +0.011 [-0.016, +0.038] | 0.475 | 0.522 | 0.529 |
| six-head, MGDA (one checkpoint) | 0.598 [0.587, 0.610] | +0.005 [-0.021, +0.029] | 0.474 | 0.508 | 0.495 |
| six-head, MGDA-GN | 0.581 [0.563, 0.599] | -0.013 [-0.041, +0.016] | 0.520 | 0.524 | 0.446 |
| six-head, MGDA-GN (one checkpoint) | 0.607 [0.593, 0.620] | +0.013 [-0.013, +0.039] | 0.488 | 0.525 | 0.496 |
| six-head, MGDA-LN | 0.562 [0.549, 0.577] | -0.031 [-0.058, -0.005] * | 0.493 | 0.499 | 0.452 |
| six-head, MGDA-LN (one checkpoint) | 0.618 [0.607, 0.630] | +0.024 [-0.001, +0.049] | 0.508 | 0.511 | 0.513 |
| six-head, MGDA-LGN | 0.608 [0.592, 0.624] | +0.014 [-0.015, +0.042] | 0.482 | 0.527 | 0.552 |
| six-head, MGDA-LGN (one checkpoint) | 0.605 [0.589, 0.621] | +0.011 [-0.016, +0.037] | 0.484 | 0.528 | 0.490 |
| six-head, UPGrad | 0.616 [0.599, 0.632] | +0.022 [-0.006, +0.050] | 0.512 | 0.519 | 0.511 |
| six-head, UPGrad (one checkpoint) | 0.602 [0.594, 0.609] | +0.009 [-0.015, +0.032] | 0.486 | 0.500 | 0.498 |

## Notes

- Median best epoch (variety / red blotch / leafroll / all six checkpoint): LS 35 / 34 / 31 / 35, MGDA 26 / 22 / 19 / 16, MGDA-GN 7 / 27 / 19 / 7, MGDA-LN 21 / 30 / 32 / 21, MGDA-LGN 8 / 18 / 10 / 8, UPGrad 32 / 36 / 22 / 32. Trainings that reached the 80-epoch limit (early stopping never triggered): LS 3 of 15, MGDA 0 of 15, MGDA-GN 0 of 15, MGDA-LN 2 of 15, MGDA-LGN 0 of 15, UPGrad 3 of 15.
- The single-task models in the red blotch columns trained on all 998 leaves. Trained on Baco, Chardonnay and Vidal only, the small ResNet without max pooling scores 0.590 (attention pool) and logistic regression 0.600 (`disease_classification/REPORT.md`).
