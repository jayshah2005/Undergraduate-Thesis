# Whole-leaf CNN on Chardonnay: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same row only (0.5 = chance). Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.464 [0.389, 0.540] | — | 0.537 |
| band z-score + mean pool | 0.534 ± 0.006 | 0.534 [0.459, 0.612] | +0.070 [-0.016, +0.159] | 0.576 |
| band z-score + attention | 0.543 ± 0.013 | 0.554 [0.476, 0.630] | +0.090 [+0.008, +0.182] | 0.568 |
| pixel SNV + mean pool | 0.530 ± 0.036 | 0.519 [0.443, 0.590] | +0.055 [-0.035, +0.140] | 0.566 |
| pixel SNV + attention | 0.488 ± 0.012 | 0.496 [0.424, 0.571] | +0.032 [-0.069, +0.134] | 0.559 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.419 | 0.521 | 0.542 | 0.200 | 0.557 |
| band z-score + mean pool | 0.539 | 0.577 | 0.447 | 0.460 | 0.614 |
| band z-score + attention | 0.560 | 0.553 | 0.558 | 0.520 | 0.567 |
| pixel SNV + mean pool | 0.588 | 0.511 | 0.503 | 0.350 | 0.583 |
| pixel SNV + attention | 0.570 | 0.542 | 0.362 | 0.330 | 0.608 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.602 [0.521, 0.679] | — | 0.605 |
| band z-score + mean pool | 0.564 ± 0.068 | 0.605 [0.533, 0.677] | +0.004 [-0.083, +0.086] | 0.632 |
| band z-score + attention | 0.616 ± 0.017 | 0.624 [0.553, 0.695] | +0.022 [-0.069, +0.109] | 0.575 |
| pixel SNV + mean pool | 0.564 ± 0.028 | 0.571 [0.491, 0.647] | -0.031 [-0.107, +0.039] | 0.563 |
| pixel SNV + attention | 0.582 ± 0.021 | 0.595 [0.518, 0.663] | -0.007 [-0.095, +0.073] | 0.574 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.673 | 0.558 | 0.638 | 0.587 | 0.532 |
| band z-score + mean pool | 0.601 | 0.587 | 0.834 | 0.482 | 0.547 |
| band z-score + attention | 0.588 | 0.587 | 0.881 | 0.569 | 0.508 |
| pixel SNV + mean pool | 0.611 | 0.582 | 0.682 | 0.413 | 0.593 |
| pixel SNV + attention | 0.617 | 0.580 | 0.863 | 0.435 | 0.497 |
