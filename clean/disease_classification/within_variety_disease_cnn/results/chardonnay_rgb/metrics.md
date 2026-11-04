# Whole-leaf CNN on Chardonnay, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.588 [0.536, 0.644] | — | — |
| CNN, mean pool | 0.531 [0.487, 0.575] | -0.057 [-0.105, -0.009] | 0.524 ± 0.004 |
| CNN, attention pool | 0.569 [0.525, 0.613] | -0.019 [-0.068, +0.034] | 0.544 ± 0.040 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| CNN, mean pool | 0.567 | 0.500 | 0.486 | 0.475 | 0.465 |
| CNN, attention pool | 0.470 | 0.598 | 0.500 | 0.525 | 0.607 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.619 [0.557, 0.679] | — | — |
| CNN, mean pool | 0.584 [0.529, 0.642] | -0.035 [-0.075, +0.002] | 0.582 ± 0.020 |
| CNN, attention pool | 0.583 [0.525, 0.641] | -0.037 [-0.078, +0.005] | 0.593 ± 0.018 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| CNN, mean pool | 0.582 | 0.533 | 0.713 | 0.573 | 0.463 |
| CNN, attention pool | 0.537 | 0.475 | 0.765 | 0.567 | 0.475 |
