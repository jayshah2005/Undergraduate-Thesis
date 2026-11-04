# Whole-leaf CNN on Vidal, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.416 [0.338, 0.493] | — | — |
| CNN, mean pool | 0.396 [0.327, 0.460] | -0.020 [-0.104, +0.069] | 0.452 ± 0.046 |
| CNN, attention pool | 0.407 [0.362, 0.464] | -0.009 [-0.078, +0.060] | 0.416 ± 0.027 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| CNN, mean pool | 0.475 | 0.398 | 0.500 | 0.315 | 0.344 |
| CNN, attention pool | 0.500 | 0.477 | 0.500 | 0.399 | 0.510 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.538 [0.474, 0.598] | — | — |
| CNN, mean pool | 0.499 [0.445, 0.550] | -0.039 [-0.091, +0.013] | 0.503 ± 0.012 |
| CNN, attention pool | 0.477 [0.435, 0.520] | -0.061 [-0.113, -0.006] | 0.465 ± 0.021 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| CNN, mean pool | 0.518 | 0.500 | 0.550 | 0.491 | 0.558 |
| CNN, attention pool | 0.547 | 0.508 | 0.475 | 0.508 | 0.529 |
