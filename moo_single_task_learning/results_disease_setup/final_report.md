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

One checkpoint for all six heads (lowest summed validation loss), as one deployed model would be:

| model | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| six-head, LS (one checkpoint) | 0.893 | 0.647 | 0.685 | 0.587 |

## Each method against LS

Paired difference in balanced accuracy [95% CI], same resampled leaves; * = interval excludes 0.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|

## Six-head against the single-task small ResNet (no max pool), attention pool

The small ResNet (no max pool), attention pool is the closest single-task model (same pooling, backbone and training). Paired, as above.

| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |
|---|---|---|---|---|
| LS − small ResNet (no max pool), attention pool | -0.012 [-0.025, +0.001] | -0.006 [-0.028, +0.014] | +0.029 [+0.007, +0.051] * | +0.030 [+0.006, +0.054] * |

## Details

### Variety, per test row

| model | bal. acc. [95% CI] | Δ vs logistic regression | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | — | 0.615 | 0.975 | 0.970 | 0.925 | 0.885 |
| small ResNet (no max pool), attention pool | 0.905 [0.888, 0.923] | +0.031 [+0.015, +0.047] * | 0.620 | 0.960 | 1.000 | 0.985 | 0.960 |
| six-head, LS | 0.893 [0.875, 0.911] | +0.019 [+0.003, +0.037] * | 0.545 | 0.985 | 0.995 | 0.975 | 0.965 |
| six-head, LS (one checkpoint) | 0.893 [0.875, 0.911] | +0.019 [+0.003, +0.037] * | 0.545 | 0.985 | 0.995 | 0.975 | 0.965 |

### Leafroll (748 leaves, 255 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.471 | 0.478 | 0.522 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.524 | 0.475 | 0.496 |
| six-head, LS | 0.596 [0.581, 0.611] | +0.024 [-0.010, +0.057] | 0.536 | 0.424 | 0.495 |
| six-head, LS (one checkpoint) | 0.647 [0.635, 0.658] | +0.075 [+0.041, +0.108] * | 0.505 | 0.500 | 0.499 |

### Red blotch, all leaves (998 leaves, 637 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | +0.027 [-0.000, +0.053] | 0.466 | 0.563 | 0.527 | 0.530 |
| six-head, LS | 0.703 [0.688, 0.717] | +0.056 [+0.031, +0.082] * | 0.517 | 0.498 | 0.529 | 0.535 |
| six-head, LS (one checkpoint) | 0.685 [0.674, 0.697] | +0.039 [+0.015, +0.063] * | 0.482 | 0.475 | 0.500 | 0.492 |

### Red blotch without Cab Franc (748 leaves, 396 with the disease), per variety

| model | bal. acc. [95% CI] | Δ vs logistic regression | Baco | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | 0.594 [0.569, 0.618] | — | 0.456 | 0.540 | 0.522 |
| small ResNet (no max pool), attention pool | 0.589 [0.567, 0.610] | -0.005 [-0.036, +0.025] | 0.466 | 0.527 | 0.530 |
| six-head, LS | 0.619 [0.601, 0.636] | +0.025 [-0.004, +0.054] | 0.517 | 0.529 | 0.535 |
| six-head, LS (one checkpoint) | 0.587 [0.575, 0.599] | -0.007 [-0.033, +0.019] | 0.482 | 0.500 | 0.492 |

## Notes

- Median best epoch (variety / red blotch / leafroll / all six checkpoint): LS 35 / 34 / 31 / 35. Trainings that reached the 80-epoch limit (early stopping never triggered): LS 3 of 15.
- The single-task models in the red blotch columns trained on all 998 leaves. Trained on Baco, Chardonnay and Vidal only, the small ResNet without max pooling scores 0.590 (attention pool) and logistic regression 0.600 (`disease_classification/REPORT.md`).
