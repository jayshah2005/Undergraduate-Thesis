# Disease logistic regression: all varieties, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). AUC compares leaves of the same variety and row only (0.5 = chance), with a 95% bootstrap interval.

| task | leaves (+ / −) | AUC [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.458 [0.410, 0.508] | 0.408 | 0.449 | 0.379 | 0.636 | 0.445 |
| red_blotch | 633 / 365 | 0.540 [0.498, 0.581] | 0.531 | 0.540 | 0.649 | 0.484 | 0.511 |
| both | 115 / 633 | 0.420 [0.363, 0.479] | 0.386 | 0.486 | 0.256 | 0.547 | 0.479 |
| any | 772 / 226 | 0.554 [0.510, 0.598] | 0.611 | 0.572 | 0.613 | 0.474 | 0.545 |

Left out (fewer than 5 positive or negative leaves): leafroll: Cab Franc; both: Cab Franc.

AUC per variety:

| task | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| leafroll | 0.417 | — | 0.483 | 0.500 |
| red_blotch | 0.418 | 0.559 | 0.632 | 0.581 |
| both | 0.442 | — | 0.366 | 0.520 |
| any | 0.463 | 0.568 | 0.588 | 0.592 |
