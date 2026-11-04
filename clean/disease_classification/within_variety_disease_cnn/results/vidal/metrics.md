# Whole-leaf CNN on Vidal, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.544 [0.491, 0.596] | — | — |
| CNN, mean pool | 0.387 [0.333, 0.442] | -0.158 [-0.231, -0.087] | 0.421 ± 0.089 |
| CNN, attention pool | 0.478 [0.460, 0.496] | -0.067 [-0.120, -0.011] | 0.449 ± 0.028 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |
| CNN, mean pool | 0.586 | 0.409 | 0.510 | 0.500 | 0.500 |
| CNN, attention pool | 0.500 | 0.477 | 0.542 | 0.500 | 0.792 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.552 [0.494, 0.610] | — | — |
| CNN, mean pool | 0.531 [0.479, 0.579] | -0.022 [-0.092, +0.050] | 0.535 ± 0.015 |
| CNN, attention pool | 0.493 [0.447, 0.537] | -0.060 [-0.133, +0.017] | 0.509 ± 0.038 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |
| CNN, mean pool | 0.635 | 0.542 | 0.575 | 0.508 | 0.553 |
| CNN, attention pool | 0.527 | 0.492 | 0.500 | 0.532 | 0.583 |
