# Whole-leaf CNN on Chardonnay: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same row only (0.5 = chance). Δ: paired difference to the baseline.

## Leafroll (122 + / 128 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.464 [0.389, 0.540] | — | 0.537 |
| band z-score + mean pool | 0.535 ± 0.002 | 0.535 [0.458, 0.619] | +0.071 [-0.026, +0.168] | 0.572 |
| band z-score + attention | 0.535 ± 0.007 | 0.546 [0.469, 0.624] | +0.082 [-0.008, +0.177] | 0.595 |
| pixel SNV + mean pool | 0.508 ± 0.006 | 0.500 [0.422, 0.573] | +0.036 [-0.058, +0.128] | 0.583 |
| pixel SNV + attention | 0.520 ± 0.011 | 0.520 [0.446, 0.597] | +0.056 [-0.028, +0.141] | 0.566 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.419 | 0.521 | 0.542 | 0.200 | 0.557 |
| band z-score + mean pool | 0.572 | 0.484 | 0.556 | 0.580 | 0.498 |
| band z-score + attention | 0.601 | 0.548 | 0.549 | 0.520 | 0.504 |
| pixel SNV + mean pool | 0.628 | 0.496 | 0.398 | 0.350 | 0.565 |
| pixel SNV + attention | 0.564 | 0.560 | 0.497 | 0.312 | 0.594 |

## Red blotch (88 + / 162 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.602 [0.521, 0.679] | — | 0.605 |
| band z-score + mean pool | 0.618 ± 0.019 | 0.640 [0.564, 0.713] | +0.038 [-0.045, +0.121] | 0.602 |
| band z-score + attention | 0.610 ± 0.034 | 0.640 [0.568, 0.709] | +0.038 [-0.045, +0.118] | 0.592 |
| pixel SNV + mean pool | 0.585 ± 0.005 | 0.588 [0.513, 0.656] | -0.014 [-0.095, +0.068] | 0.613 |
| pixel SNV + attention | 0.497 ± 0.062 | 0.491 [0.412, 0.572] | -0.110 [-0.196, -0.024] | 0.511 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.673 | 0.558 | 0.638 | 0.587 | 0.532 |
| band z-score + mean pool | 0.638 | 0.590 | 0.859 | 0.596 | 0.517 |
| band z-score + attention | 0.623 | 0.585 | 0.879 | 0.569 | 0.562 |
| pixel SNV + mean pool | 0.607 | 0.588 | 0.875 | 0.392 | 0.497 |
| pixel SNV + attention | 0.590 | 0.542 | 0.322 | 0.424 | 0.565 |
