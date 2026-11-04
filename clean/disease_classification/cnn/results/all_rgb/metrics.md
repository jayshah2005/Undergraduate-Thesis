# Whole-leaf CNN on all varieties, RGB bands only: metrics

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

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.641 [0.619, 0.664] | — | — |
| CNN, mean pool | 0.650 [0.631, 0.668] | +0.009 [-0.012, +0.030] | 0.663 ± 0.012 |
| CNN, attention pool | 0.667 [0.647, 0.686] | +0.026 [+0.003, +0.048] | 0.666 ± 0.007 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| CNN, mean pool | 0.636 | 0.635 | 0.651 | 0.679 | 0.657 |
| CNN, attention pool | 0.659 | 0.647 | 0.666 | 0.688 | 0.674 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.458 | 0.499 | 0.567 | 0.542 |
| CNN, mean pool | 0.497 | 0.489 | 0.517 | 0.471 |
| CNN, attention pool | 0.503 | 0.502 | 0.534 | 0.503 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs baseline [95% CI] | single seeds (mean ± SD) |
|---|---|---|---|
| baseline | 0.584 [0.558, 0.612] | — | — |
| CNN, mean pool | 0.614 [0.592, 0.635] | +0.030 [+0.001, +0.056] | 0.611 ± 0.006 |
| CNN, attention pool | 0.597 [0.575, 0.619] | +0.013 [-0.016, +0.041] | 0.600 ± 0.002 |

Balanced accuracy per test row:

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| CNN, mean pool | 0.527 | 0.567 | 0.606 | 0.645 | 0.651 |
| CNN, attention pool | 0.491 | 0.543 | 0.606 | 0.627 | 0.618 |

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.449 | 0.495 | 0.542 | 0.541 |
| CNN, mean pool | 0.487 | 0.563 | 0.500 | 0.536 |
| CNN, attention pool | 0.521 | 0.538 | 0.500 | 0.493 |
