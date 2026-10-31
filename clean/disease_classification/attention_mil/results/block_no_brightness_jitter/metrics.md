# Whole-leaf CNN on Chardonnay: metrics, `block` folds

AUC compares leaves of the same fold and row only (0.5 = chance). Δ: paired difference to the 4-fold baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.433 [0.355, 0.509] | — | 0.508 |
| baseline (3 folds) | — | 0.496 [0.418, 0.574] | +0.063 [+0.011, +0.117] | 0.504 |
| band z-score + mean pool | 0.519 ± 0.026 | 0.528 [0.448, 0.608] | +0.095 [-0.002, +0.193] | 0.492 |
| band z-score + attention | 0.499 ± 0.028 | 0.513 [0.433, 0.591] | +0.080 [-0.019, +0.184] | 0.458 |
| pixel SNV + mean pool | 0.460 ± 0.046 | 0.459 [0.381, 0.541] | +0.026 [-0.078, +0.128] | 0.476 |
| pixel SNV + attention | 0.497 ± 0.040 | 0.496 [0.411, 0.580] | +0.063 [-0.032, +0.154] | 0.510 |

AUC per test fold (ensemble):

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.367 | 0.387 | 0.385 | 0.451 | 0.627 |
| baseline (3 folds) | 0.500 | 0.484 | 0.475 | 0.537 | 0.493 |
| band z-score + mean pool | 0.767 | 0.237 | 0.541 | 0.561 | 0.547 |
| band z-score + attention | 0.711 | 0.333 | 0.475 | 0.512 | 0.560 |
| pixel SNV + mean pool | 0.367 | 0.290 | 0.459 | 0.720 | 0.493 |
| pixel SNV + attention | 0.533 | 0.333 | 0.484 | 0.585 | 0.573 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.577 [0.498, 0.660] | — | 0.595 |
| baseline (3 folds) | — | 0.547 [0.466, 0.628] | -0.029 [-0.092, +0.034] | 0.584 |
| band z-score + mean pool | 0.572 ± 0.028 | 0.572 [0.493, 0.655] | -0.005 [-0.101, +0.095] | 0.564 |
| band z-score + attention | 0.574 ± 0.007 | 0.559 [0.477, 0.644] | -0.018 [-0.115, +0.079] | 0.578 |
| pixel SNV + mean pool | 0.526 ± 0.028 | 0.559 [0.475, 0.640] | -0.018 [-0.110, +0.077] | 0.538 |
| pixel SNV + attention | 0.547 ± 0.024 | 0.536 [0.457, 0.615] | -0.041 [-0.126, +0.041] | 0.546 |

AUC per test fold (ensemble):

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.579 | 0.515 | 0.636 | 0.534 | 0.591 |
| baseline (3 folds) | 0.544 | 0.500 | 0.509 | 0.602 | 0.591 |
| band z-score + mean pool | 0.702 | 0.561 | 0.518 | 0.511 | 0.530 |
| band z-score + attention | 0.719 | 0.530 | 0.518 | 0.489 | 0.470 |
| pixel SNV + mean pool | 0.667 | 0.682 | 0.518 | 0.466 | 0.439 |
| pixel SNV + attention | 0.596 | 0.667 | 0.455 | 0.568 | 0.394 |
