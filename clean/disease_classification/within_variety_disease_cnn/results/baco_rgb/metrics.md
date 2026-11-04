# Whole-leaf CNN on Baco, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.418 [0.371, 0.465] | — | — |
| CNN, mean pool | 0.370 [0.344, 0.398] | -0.049 [-0.097, +0.002] | 0.401 ± 0.035 |
| CNN, attention pool | 0.440 [0.407, 0.472] | +0.022 [-0.028, +0.074] | 0.439 ± 0.070 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| CNN, mean pool | 0.500 | 0.519 | 0.304 | 0.400 | 0.517 |
| CNN, attention pool | 0.500 | 0.500 | 0.357 | 0.425 | 0.543 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.543 [0.494, 0.595] | — | — |
| CNN, mean pool | 0.525 [0.475, 0.576] | -0.018 [-0.077, +0.037] | 0.527 ± 0.020 |
| CNN, attention pool | 0.548 [0.500, 0.595] | +0.005 [-0.060, +0.071] | 0.549 ± 0.009 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| CNN, mean pool | 0.649 | 0.524 | 0.502 | 0.500 | 0.397 |
| CNN, attention pool | 0.646 | 0.513 | 0.554 | 0.500 | 0.449 |
