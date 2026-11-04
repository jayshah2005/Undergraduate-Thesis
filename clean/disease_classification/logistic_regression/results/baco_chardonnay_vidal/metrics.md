# Disease logistic regression: Baco + Chardonnay + Vidal, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.577 [0.545, 0.610] | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| red_blotch | 392 / 356 | 0.597 [0.570, 0.624] | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| both | 115 / 633 | 0.533 [0.492, 0.576] | 0.508 | 0.495 | 0.678 | 0.428 | 0.522 |
| any | 531 / 217 | 0.523 [0.488, 0.558] | 0.583 | 0.516 | 0.586 | 0.512 | 0.464 |

Balanced accuracy per variety:

| task | Baco | Chardonnay | Vidal |
|---|---|---|---|
| leafroll | 0.470 | 0.482 | 0.527 |
| red_blotch | 0.460 | 0.548 | 0.546 |
| both | 0.445 | 0.535 | 0.466 |
| any | 0.446 | 0.494 | 0.580 |
