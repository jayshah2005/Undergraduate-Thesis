# Whole-leaf CNN on Baco, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.524 [0.464, 0.587] | — | — |
| CNN, mean pool | 0.546 [0.522, 0.571] | +0.021 [-0.043, +0.084] | 0.550 ± 0.029 |
| CNN, attention pool | 0.529 [0.502, 0.556] | +0.005 [-0.060, +0.068] | 0.546 ± 0.037 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| CNN, mean pool | 0.524 | 0.500 | 0.525 | 0.500 | 0.500 |
| CNN, attention pool | 0.500 | 0.494 | 0.500 | 0.463 | 0.500 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.506 [0.445, 0.565] | — | — |
| CNN, mean pool | 0.504 [0.456, 0.550] | -0.001 [-0.074, +0.074] | 0.486 ± 0.013 |
| CNN, attention pool | 0.505 [0.465, 0.547] | -0.001 [-0.068, +0.069] | 0.516 ± 0.006 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| CNN, mean pool | 0.519 | 0.500 | 0.451 | 0.600 | 0.471 |
| CNN, attention pool | 0.552 | 0.409 | 0.500 | 0.500 | 0.441 |
