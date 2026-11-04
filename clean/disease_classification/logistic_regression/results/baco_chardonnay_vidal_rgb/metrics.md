# Disease logistic regression: Baco + Chardonnay + Vidal, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.544 [0.513, 0.575] | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| red_blotch | 392 / 356 | 0.604 [0.579, 0.630] | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| both | 115 / 633 | 0.492 [0.456, 0.529] | 0.503 | 0.527 | 0.470 | 0.511 | 0.407 |
| any | 531 / 217 | 0.564 [0.535, 0.592] | 0.574 | 0.510 | 0.662 | 0.528 | 0.584 |

Balanced accuracy per variety:

| task | Baco | Chardonnay | Vidal |
|---|---|---|---|
| leafroll | 0.459 | 0.428 | 0.469 |
| red_blotch | 0.434 | 0.610 | 0.551 |
| both | 0.459 | 0.405 | 0.553 |
| any | 0.427 | 0.610 | 0.553 |
