# Whole-leaf CNN on Chardonnay: metrics, `photo` folds

AUC compares leaves of the same fold and row only (0.5 = chance). Δ: paired difference to the 4-fold baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.489 [0.408, 0.572] | — | 0.562 |
| baseline (3 folds) | — | 0.515 [0.437, 0.596] | +0.026 [-0.033, +0.085] | 0.530 |
| band z-score + mean pool | 0.468 ± 0.025 | 0.463 [0.386, 0.544] | -0.026 [-0.129, +0.072] | 0.491 |
| band z-score + attention | 0.489 ± 0.020 | 0.476 [0.400, 0.550] | -0.013 [-0.111, +0.081] | 0.502 |
| pixel SNV + mean pool | 0.558 ± 0.003 | 0.594 [0.522, 0.670] | +0.105 [+0.004, +0.203] | 0.514 |
| pixel SNV + attention | 0.552 ± 0.028 | 0.568 [0.489, 0.646] | +0.079 [-0.031, +0.183] | 0.485 |

AUC per test fold (ensemble):

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.540 | 0.510 | 0.449 | 0.459 | 0.466 |
| baseline (3 folds) | 0.602 | 0.510 | 0.398 | 0.568 | 0.493 |
| band z-score + mean pool | 0.345 | 0.530 | 0.592 | 0.378 | 0.466 |
| band z-score + attention | 0.469 | 0.620 | 0.602 | 0.297 | 0.301 |
| pixel SNV + mean pool | 0.584 | 0.580 | 0.582 | 0.608 | 0.630 |
| pixel SNV + attention | 0.487 | 0.620 | 0.684 | 0.459 | 0.575 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.604 [0.533, 0.671] | — | 0.606 |
| baseline (3 folds) | — | 0.579 [0.504, 0.650] | -0.024 [-0.079, +0.026] | 0.615 |
| band z-score + mean pool | 0.592 ± 0.009 | 0.587 [0.514, 0.654] | -0.016 [-0.100, +0.067] | 0.594 |
| band z-score + attention | 0.604 ± 0.012 | 0.614 [0.541, 0.687] | +0.010 [-0.069, +0.089] | 0.605 |
| pixel SNV + mean pool | 0.577 ± 0.001 | 0.591 [0.526, 0.659] | -0.012 [-0.091, +0.067] | 0.562 |
| pixel SNV + attention | 0.606 ± 0.022 | 0.620 [0.553, 0.687] | +0.016 [-0.061, +0.091] | 0.597 |

AUC per test fold (ensemble):

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.394 | 0.718 | 0.510 | 0.731 | 0.663 |
| baseline (3 folds) | 0.414 | 0.757 | 0.470 | 0.663 | 0.581 |
| band z-score + mean pool | 0.434 | 0.524 | 0.660 | 0.644 | 0.686 |
| band z-score + attention | 0.525 | 0.515 | 0.640 | 0.673 | 0.733 |
| pixel SNV + mean pool | 0.475 | 0.621 | 0.520 | 0.663 | 0.686 |
| pixel SNV + attention | 0.444 | 0.660 | 0.570 | 0.663 | 0.779 |
