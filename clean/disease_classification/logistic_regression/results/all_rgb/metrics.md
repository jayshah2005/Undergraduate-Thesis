# Disease logistic regression: all varieties, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 254 / 494 | 0.544 [0.513, 0.575] | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| red_blotch | 633 / 365 | 0.641 [0.619, 0.664] | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| both | 115 / 633 | 0.492 [0.456, 0.529] | 0.503 | 0.527 | 0.470 | 0.511 | 0.407 |
| any | 772 / 226 | 0.584 [0.558, 0.612] | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |

Left out (fewer than 5 positive or negative leaves): leafroll: Cab Franc; both: Cab Franc.

Balanced accuracy per variety:

| task | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| leafroll | 0.459 | — | 0.428 | 0.469 |
| red_blotch | 0.458 | 0.499 | 0.567 | 0.542 |
| both | 0.459 | — | 0.405 | 0.553 |
| any | 0.449 | 0.495 | 0.542 | 0.541 |
