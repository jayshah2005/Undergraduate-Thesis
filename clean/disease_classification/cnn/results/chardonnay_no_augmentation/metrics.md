# Whole-leaf CNN on Chardonnay, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.537 [0.477, 0.596] | — | — |
| band z-score + mean pool | 0.576 [0.524, 0.628] | +0.040 [-0.029, +0.111] | 0.559 ± 0.023 |
| band z-score + attention | 0.568 [0.520, 0.620] | +0.032 [-0.038, +0.104] | 0.563 ± 0.005 |
| pixel SNV + mean pool | 0.566 [0.543, 0.590] | +0.030 [-0.035, +0.093] | 0.568 ± 0.016 |
| pixel SNV + attention | 0.559 [0.526, 0.591] | +0.022 [-0.046, +0.086] | 0.545 ± 0.012 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| band z-score + mean pool | 0.535 | 0.520 | 0.476 | 0.487 | 0.575 |
| band z-score + attention | 0.572 | 0.494 | 0.562 | 0.475 | 0.530 |
| pixel SNV + mean pool | 0.522 | 0.500 | 0.471 | 0.487 | 0.502 |
| pixel SNV + attention | 0.500 | 0.504 | 0.500 | 0.463 | 0.539 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.605 [0.547, 0.662] | — | — |
| band z-score + mean pool | 0.632 [0.579, 0.682] | +0.027 [-0.040, +0.092] | 0.573 ± 0.057 |
| band z-score + attention | 0.575 [0.516, 0.634] | -0.030 [-0.100, +0.038] | 0.580 ± 0.039 |
| pixel SNV + mean pool | 0.563 [0.510, 0.615] | -0.042 [-0.100, +0.016] | 0.578 ± 0.003 |
| pixel SNV + attention | 0.574 [0.519, 0.624] | -0.031 [-0.097, +0.031] | 0.566 ± 0.047 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| band z-score + mean pool | 0.604 | 0.542 | 0.754 | 0.551 | 0.562 |
| band z-score + attention | 0.473 | 0.575 | 0.738 | 0.561 | 0.512 |
| pixel SNV + mean pool | 0.514 | 0.542 | 0.654 | 0.510 | 0.550 |
| pixel SNV + attention | 0.529 | 0.525 | 0.779 | 0.500 | 0.500 |
