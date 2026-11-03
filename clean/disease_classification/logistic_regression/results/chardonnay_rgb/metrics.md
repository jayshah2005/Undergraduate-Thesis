# Disease logistic regression: Chardonnay, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 122 / 128 | 0.588 [0.536, 0.644] | 0.446 | 0.663 | 0.668 | 0.512 | 0.600 |
| red_blotch | 88 / 162 | 0.619 [0.557, 0.679] | 0.540 | 0.500 | 0.779 | 0.660 | 0.575 |
| both | 50 / 200 | 0.595 [0.525, 0.665] | 0.493 | 0.688 | 0.658 | 0.337 | 0.560 |
| any | 160 / 90 | 0.573 [0.514, 0.631] | 0.500 | 0.445 | 0.594 | 0.660 | 0.539 |
