# Disease logistic regression: all varieties, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). AUC compares leaves of the same variety and row only (0.5 = chance), with a 95% bootstrap interval.

| task | leaves (+ / −) | AUC [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.507 [0.460, 0.553] | 0.491 | 0.523 | 0.597 | 0.441 | 0.482 |
| red_blotch | 633 / 365 | 0.517 [0.476, 0.560] | 0.459 | 0.610 | 0.564 | 0.452 | 0.507 |
| both | 115 / 633 | 0.479 [0.422, 0.533] | 0.416 | 0.439 | 0.626 | 0.478 | 0.464 |
| any | 772 / 226 | 0.523 [0.478, 0.566] | 0.460 | 0.519 | 0.620 | 0.491 | 0.523 |

Left out (fewer than 5 positive or negative leaves): leafroll: Cab Franc; both: Cab Franc.

AUC per variety:

| task | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| leafroll | 0.485 | — | 0.505 | 0.569 |
| red_blotch | 0.417 | 0.469 | 0.592 | 0.562 |
| both | 0.426 | — | 0.496 | 0.609 |
| any | 0.460 | 0.378 | 0.548 | 0.574 |
