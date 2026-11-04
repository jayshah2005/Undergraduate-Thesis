# Whole-leaf CNN on Chardonnay, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.588 [0.536, 0.644] | — | — |
| CNN, mean pool | 0.518 [0.471, 0.566] | -0.070 [-0.117, -0.026] | 0.514 ± 0.030 |
| CNN, attention pool | 0.566 [0.522, 0.609] | -0.022 [-0.074, +0.030] | 0.546 ± 0.029 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN, mean pool | 0.567 | 0.488 | 0.471 | 0.450 | 0.493 |
| CNN, attention pool | 0.517 | 0.520 | 0.471 | 0.550 | 0.562 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.619 [0.557, 0.679] | — | — |
| CNN, mean pool | 0.604 [0.543, 0.662] | -0.016 [-0.054, +0.022] | 0.585 ± 0.026 |
| CNN, attention pool | 0.582 [0.528, 0.636] | -0.038 [-0.081, +0.007] | 0.591 ± 0.018 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN, mean pool | 0.582 | 0.533 | 0.727 | 0.593 | 0.562 |
| CNN, attention pool | 0.537 | 0.508 | 0.740 | 0.553 | 0.475 |
