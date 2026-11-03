# Whole-leaf CNN on Chardonnay, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.588 [0.536, 0.644] | — | — |
| band z-score + mean pool | 0.542 [0.494, 0.590] | -0.046 [-0.106, +0.011] | 0.524 ± 0.026 |
| band z-score + attention | 0.572 [0.524, 0.620] | -0.016 [-0.072, +0.036] | 0.566 ± 0.006 |
| pixel SNV + mean pool | 0.518 [0.471, 0.566] | -0.070 [-0.117, -0.026] | 0.514 ± 0.030 |
| pixel SNV + attention | 0.566 [0.522, 0.609] | -0.022 [-0.074, +0.030] | 0.546 ± 0.029 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| band z-score + mean pool | 0.523 | 0.452 | 0.592 | 0.500 | 0.536 |
| band z-score + attention | 0.548 | 0.484 | 0.561 | 0.512 | 0.568 |
| pixel SNV + mean pool | 0.567 | 0.488 | 0.471 | 0.450 | 0.493 |
| pixel SNV + attention | 0.517 | 0.520 | 0.471 | 0.550 | 0.562 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.619 [0.557, 0.679] | — | — |
| band z-score + mean pool | 0.582 [0.522, 0.642] | -0.038 [-0.093, +0.017] | 0.565 ± 0.029 |
| band z-score + attention | 0.625 [0.567, 0.681] | +0.006 [-0.053, +0.059] | 0.599 ± 0.031 |
| pixel SNV + mean pool | 0.604 [0.543, 0.662] | -0.016 [-0.054, +0.022] | 0.585 ± 0.026 |
| pixel SNV + attention | 0.582 [0.528, 0.636] | -0.038 [-0.081, +0.007] | 0.591 ± 0.018 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| band z-score + mean pool | 0.542 | 0.567 | 0.749 | 0.525 | 0.538 |
| band z-score + attention | 0.535 | 0.583 | 0.815 | 0.551 | 0.588 |
| pixel SNV + mean pool | 0.582 | 0.533 | 0.727 | 0.593 | 0.562 |
| pixel SNV + attention | 0.537 | 0.508 | 0.740 | 0.553 | 0.475 |
