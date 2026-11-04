# Whole-leaf CNN on Chardonnay, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.537 [0.477, 0.596] | — | — |
| CNN, mean pool | 0.591 [0.555, 0.628] | +0.055 [-0.014, +0.119] | 0.558 ± 0.016 |
| CNN, attention pool | 0.559 [0.531, 0.587] | +0.023 [-0.038, +0.079] | 0.554 ± 0.021 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| CNN, mean pool | 0.615 | 0.500 | 0.501 | 0.475 | 0.572 |
| CNN, attention pool | 0.500 | 0.500 | 0.471 | 0.475 | 0.522 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.605 [0.547, 0.662] | — | — |
| CNN, mean pool | 0.523 [0.475, 0.570] | -0.082 [-0.143, -0.020] | 0.541 ± 0.022 |
| CNN, attention pool | 0.519 [0.472, 0.566] | -0.086 [-0.152, -0.023] | 0.514 ± 0.024 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| CNN, mean pool | 0.514 | 0.575 | 0.627 | 0.494 | 0.500 |
| CNN, attention pool | 0.537 | 0.600 | 0.668 | 0.500 | 0.500 |
