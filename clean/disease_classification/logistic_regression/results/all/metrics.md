# Disease logistic regression: all varieties, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.577 [0.545, 0.610] | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| red_blotch | 633 / 365 | 0.637 [0.616, 0.659] | 0.531 | 0.675 | 0.700 | 0.619 | 0.673 |
| both | 115 / 633 | 0.533 [0.492, 0.576] | 0.508 | 0.495 | 0.678 | 0.428 | 0.522 |
| any | 772 / 226 | 0.556 [0.523, 0.587] | 0.420 | 0.615 | 0.653 | 0.517 | 0.605 |

Left out (fewer than 5 positive or negative leaves): leafroll: Cab Franc; both: Cab Franc.

Balanced accuracy per variety:

| task | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| leafroll | 0.470 | — | 0.482 | 0.527 |
| red_blotch | 0.437 | 0.473 | 0.549 | 0.515 |
| both | 0.445 | — | 0.535 | 0.466 |
| any | 0.431 | 0.435 | 0.491 | 0.609 |
