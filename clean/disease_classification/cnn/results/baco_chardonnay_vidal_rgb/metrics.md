# Whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. Balanced accuracy over all test rows (0.5 = chance); a leaf counts as positive at P ≥ 0.5. CNN = mean P of 3 seeds. Δ: paired difference to the baseline. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.544 [0.513, 0.575] | — | — |
| CNN, mean pool | 0.565 [0.535, 0.593] | +0.021 [-0.004, +0.045] | 0.561 ± 0.015 |
| CNN, attention pool | 0.580 [0.552, 0.607] | +0.036 [+0.010, +0.064] | 0.576 ± 0.011 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| CNN, mean pool | 0.557 | 0.627 | 0.546 | 0.546 | 0.578 |
| CNN, attention pool | 0.583 | 0.627 | 0.568 | 0.559 | 0.574 |

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| baseline | 0.459 | 0.428 | 0.469 |
| CNN, mean pool | 0.462 | 0.436 | 0.493 |
| CNN, attention pool | 0.540 | 0.423 | 0.502 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.604 [0.579, 0.630] | — | — |
| CNN, mean pool | 0.602 [0.576, 0.628] | -0.003 [-0.021, +0.018] | 0.587 ± 0.010 |
| CNN, attention pool | 0.607 [0.582, 0.633] | +0.003 [-0.015, +0.022] | 0.596 ± 0.007 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| CNN, mean pool | 0.612 | 0.567 | 0.673 | 0.550 | 0.641 |
| CNN, attention pool | 0.612 | 0.553 | 0.649 | 0.597 | 0.662 |

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| baseline | 0.434 | 0.610 | 0.551 |
| CNN, mean pool | 0.455 | 0.591 | 0.546 |
| CNN, attention pool | 0.483 | 0.565 | 0.562 |
