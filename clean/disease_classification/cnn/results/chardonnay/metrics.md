# Whole-leaf CNN on Chardonnay, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.537 [0.477, 0.596] | — | — |
| band z-score + mean pool | 0.572 [0.525, 0.620] | +0.036 [-0.036, +0.111] | 0.579 ± 0.001 |
| band z-score + attention | 0.595 [0.544, 0.647] | +0.059 [-0.011, +0.132] | 0.584 ± 0.011 |
| pixel SNV + mean pool | 0.583 [0.551, 0.619] | +0.047 [-0.022, +0.111] | 0.557 ± 0.007 |
| pixel SNV + attention | 0.566 [0.525, 0.605] | +0.029 [-0.040, +0.094] | 0.527 ± 0.040 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| band z-score + mean pool | 0.578 | 0.468 | 0.489 | 0.513 | 0.484 |
| band z-score + attention | 0.566 | 0.520 | 0.550 | 0.475 | 0.501 |
| pixel SNV + mean pool | 0.590 | 0.500 | 0.471 | 0.487 | 0.572 |
| pixel SNV + attention | 0.543 | 0.500 | 0.443 | 0.462 | 0.609 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.605 [0.547, 0.662] | — | — |
| band z-score + mean pool | 0.602 [0.547, 0.657] | -0.003 [-0.070, +0.061] | 0.577 ± 0.018 |
| band z-score + attention | 0.592 [0.533, 0.648] | -0.013 [-0.077, +0.049] | 0.581 ± 0.021 |
| pixel SNV + mean pool | 0.613 [0.562, 0.662] | +0.008 [-0.055, +0.068] | 0.553 ± 0.039 |
| pixel SNV + attention | 0.511 [0.470, 0.553] | -0.094 [-0.158, -0.033] | 0.508 ± 0.020 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| band z-score + mean pool | 0.569 | 0.550 | 0.738 | 0.573 | 0.500 |
| band z-score + attention | 0.492 | 0.533 | 0.801 | 0.557 | 0.550 |
| pixel SNV + mean pool | 0.514 | 0.600 | 0.871 | 0.526 | 0.512 |
| pixel SNV + attention | 0.537 | 0.583 | 0.500 | 0.500 | 0.512 |
