# Disease logistic regression: Chardonnay, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). AUC compares leaves of the same row only (0.5 = chance), with a 95% bootstrap interval.

| task | leaves (+ / −) | AUC [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 122 / 128 | 0.510 [0.436, 0.587] | 0.401 | 0.570 | 0.579 | 0.450 | 0.537 |
| red_blotch | 88 / 162 | 0.626 [0.551, 0.697] | 0.599 | 0.518 | 0.827 | 0.655 | 0.542 |
| both | 50 / 200 | 0.615 [0.531, 0.699] | 0.574 | 0.692 | 0.743 | 0.174 | 0.663 |
| any | 160 / 90 | 0.603 [0.531, 0.672] | 0.587 | 0.456 | 0.588 | 0.669 | 0.668 |
