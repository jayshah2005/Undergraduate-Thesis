# Disease logistic regression: Chardonnay, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). AUC compares leaves of the same row only (0.5 = chance), with a 95% bootstrap interval.

| task | leaves (+ / −) | AUC [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 122 / 128 | 0.464 [0.389, 0.540] | 0.419 | 0.521 | 0.542 | 0.200 | 0.557 |
| red_blotch | 88 / 162 | 0.602 [0.521, 0.679] | 0.673 | 0.558 | 0.638 | 0.587 | 0.532 |
| both | 50 / 200 | 0.439 [0.350, 0.533] | 0.299 | 0.505 | 0.599 | 0.223 | 0.516 |
| any | 160 / 90 | 0.575 [0.501, 0.651] | 0.552 | 0.518 | 0.722 | 0.520 | 0.562 |
