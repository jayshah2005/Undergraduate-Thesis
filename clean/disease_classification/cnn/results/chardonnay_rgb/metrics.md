# Whole-leaf CNN on Chardonnay, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same row only (0.5 = chance). Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.510 [0.436, 0.587] | — | 0.588 |
| band z-score + mean pool | 0.525 ± 0.037 | 0.517 [0.443, 0.595] | +0.008 [-0.083, +0.103] | 0.542 |
| band z-score + attention | 0.529 ± 0.005 | 0.548 [0.473, 0.623] | +0.038 [-0.043, +0.120] | 0.572 |
| pixel SNV + mean pool | 0.542 ± 0.021 | 0.559 [0.487, 0.633] | +0.050 [-0.020, +0.121] | 0.518 |
| pixel SNV + attention | 0.564 ± 0.017 | 0.583 [0.513, 0.656] | +0.074 [-0.000, +0.147] | 0.566 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.401 | 0.570 | 0.579 | 0.450 | 0.537 |
| band z-score + mean pool | 0.519 | 0.543 | 0.492 | 0.453 | 0.557 |
| band z-score + attention | 0.498 | 0.698 | 0.535 | 0.390 | 0.570 |
| pixel SNV + mean pool | 0.580 | 0.626 | 0.545 | 0.460 | 0.552 |
| pixel SNV + attention | 0.549 | 0.759 | 0.512 | 0.495 | 0.573 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.626 [0.551, 0.697] | — | 0.619 |
| band z-score + mean pool | 0.602 ± 0.010 | 0.613 [0.537, 0.687] | -0.013 [-0.064, +0.038] | 0.582 |
| band z-score + attention | 0.608 ± 0.012 | 0.621 [0.544, 0.694] | -0.004 [-0.060, +0.048] | 0.625 |
| pixel SNV + mean pool | 0.585 ± 0.034 | 0.608 [0.530, 0.682] | -0.017 [-0.049, +0.013] | 0.604 |
| pixel SNV + attention | 0.603 ± 0.013 | 0.613 [0.535, 0.686] | -0.013 [-0.045, +0.016] | 0.582 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.599 | 0.518 | 0.827 | 0.655 | 0.542 |
| band z-score + mean pool | 0.595 | 0.577 | 0.813 | 0.537 | 0.570 |
| band z-score + attention | 0.607 | 0.563 | 0.819 | 0.553 | 0.593 |
| pixel SNV + mean pool | 0.604 | 0.547 | 0.755 | 0.608 | 0.532 |
| pixel SNV + attention | 0.606 | 0.547 | 0.786 | 0.591 | 0.547 |
