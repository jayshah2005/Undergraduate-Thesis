# Whole-leaf CNN on all varieties, RGB bands only: metrics

Each row is tested once; the next row is validation and the other 3 train, for the CNN and the baseline. AUC compares leaves of the same variety and row only (0.5 = chance). Δ: paired difference to the baseline. Bal. acc. isn't compared within a variety, so recognising a variety can raise it.

## Any disease vs healthy (772 + / 226 −)

| model | AUC seeds (mean ± SD) | AUC ensemble [95% CI] | Δ vs baseline [95% CI] | bal. acc. |
|---|---|---|---|---|
| baseline | — | 0.554 [0.510, 0.598] | — | 0.584 |
| band z-score + mean pool | 0.499 ± 0.027 | 0.513 [0.467, 0.558] | -0.041 [-0.087, +0.005] | 0.595 |
| band z-score + attention | 0.529 ± 0.009 | 0.528 [0.484, 0.573] | -0.026 [-0.076, +0.024] | 0.592 |
| pixel SNV + mean pool | 0.531 ± 0.003 | 0.521 [0.476, 0.566] | -0.034 [-0.081, +0.013] | 0.614 |
| pixel SNV + attention | 0.521 ± 0.012 | 0.528 [0.485, 0.573] | -0.026 [-0.076, +0.024] | 0.597 |

AUC per test row (ensemble):

| model | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| baseline | 0.611 | 0.572 | 0.613 | 0.474 | 0.545 |
| band z-score + mean pool | 0.586 | 0.513 | 0.626 | 0.393 | 0.507 |
| band z-score + attention | 0.611 | 0.546 | 0.634 | 0.393 | 0.522 |
| pixel SNV + mean pool | 0.544 | 0.552 | 0.522 | 0.454 | 0.554 |
| pixel SNV + attention | 0.543 | 0.555 | 0.527 | 0.452 | 0.583 |

AUC per variety (ensemble):

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| baseline | 0.463 | 0.568 | 0.588 | 0.592 |
| band z-score + mean pool | 0.431 | 0.550 | 0.539 | 0.547 |
| band z-score + attention | 0.435 | 0.524 | 0.577 | 0.551 |
| pixel SNV + mean pool | 0.426 | 0.536 | 0.553 | 0.562 |
| pixel SNV + attention | 0.451 | 0.615 | 0.571 | 0.530 |
