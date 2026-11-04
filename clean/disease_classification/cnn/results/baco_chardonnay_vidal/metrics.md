# Whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.577 [0.545, 0.610] | — | — |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.611 ± 0.019 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.610 ± 0.002 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| baseline | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.597 [0.570, 0.624] | — | — |
| CNN, mean pool | 0.608 [0.585, 0.631] | +0.012 [-0.019, +0.043] | 0.600 ± 0.010 |
| CNN, attention pool | 0.605 [0.583, 0.626] | +0.008 [-0.022, +0.037] | 0.590 ± 0.023 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| CNN, mean pool | 0.598 | 0.589 | 0.609 | 0.569 | 0.653 |
| CNN, attention pool | 0.559 | 0.579 | 0.681 | 0.571 | 0.632 |

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| baseline | 0.460 | 0.548 | 0.546 |
| CNN, mean pool | 0.483 | 0.565 | 0.559 |
| CNN, attention pool | 0.452 | 0.577 | 0.562 |
