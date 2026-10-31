# Whole-leaf CNN on Chardonnay: metrics, `row` folds

AUC compares leaves of the same fold and row only (0.5 = chance). Δ: paired difference to the 4-fold baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.460 [0.387, 0.538] | — | 0.503 |
| baseline (3 folds) | — | 0.464 [0.389, 0.540] | +0.004 [-0.050, +0.055] | 0.537 |
| band z-score + mean pool | 0.535 ± 0.002 | 0.535 [0.458, 0.619] | +0.074 [-0.020, +0.169] | 0.572 |
| band z-score + attention | 0.535 ± 0.007 | 0.546 [0.469, 0.624] | +0.086 [-0.007, +0.177] | 0.595 |
| pixel SNV + mean pool | 0.508 ± 0.006 | 0.500 [0.422, 0.573] | +0.039 [-0.052, +0.125] | 0.583 |
| pixel SNV + attention | 0.520 ± 0.011 | 0.520 [0.446, 0.597] | +0.060 [-0.019, +0.139] | 0.566 |

AUC per test fold (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.374 | 0.538 | 0.540 | 0.295 | 0.509 |
| baseline (3 folds) | 0.419 | 0.521 | 0.542 | 0.200 | 0.557 |
| band z-score + mean pool | 0.572 | 0.484 | 0.556 | 0.580 | 0.498 |
| band z-score + attention | 0.601 | 0.548 | 0.549 | 0.520 | 0.504 |
| pixel SNV + mean pool | 0.628 | 0.496 | 0.398 | 0.350 | 0.565 |
| pixel SNV + attention | 0.564 | 0.560 | 0.497 | 0.312 | 0.594 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline (4 folds) | — | 0.565 [0.482, 0.646] | — | 0.582 |
| baseline (3 folds) | — | 0.602 [0.521, 0.679] | +0.036 [-0.019, +0.096] | 0.605 |
| band z-score + mean pool | 0.618 ± 0.019 | 0.640 [0.564, 0.713] | +0.074 [-0.007, +0.162] | 0.602 |
| band z-score + attention | 0.610 ± 0.034 | 0.640 [0.568, 0.709] | +0.074 [-0.013, +0.160] | 0.592 |
| pixel SNV + mean pool | 0.585 ± 0.005 | 0.588 [0.513, 0.656] | +0.022 [-0.061, +0.106] | 0.613 |
| pixel SNV + attention | 0.497 ± 0.062 | 0.491 [0.412, 0.572] | -0.074 [-0.160, +0.016] | 0.511 |

AUC per test fold (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline (4 folds) | 0.577 | 0.543 | 0.568 | 0.582 | 0.552 |
| baseline (3 folds) | 0.673 | 0.558 | 0.638 | 0.587 | 0.532 |
| band z-score + mean pool | 0.638 | 0.590 | 0.859 | 0.596 | 0.517 |
| band z-score + attention | 0.623 | 0.585 | 0.879 | 0.569 | 0.562 |
| pixel SNV + mean pool | 0.607 | 0.588 | 0.875 | 0.392 | 0.497 |
| pixel SNV + attention | 0.590 | 0.542 | 0.322 | 0.424 | 0.565 |
