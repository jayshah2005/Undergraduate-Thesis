# Disease logistic regression: Baco, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 107 / 141 | 0.524 [0.464, 0.587] | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| red_blotch | 135 / 113 | 0.506 [0.445, 0.565] | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| both | 53 / 195 | 0.470 [0.401, 0.536] | 0.413 | 0.559 | 0.418 | 0.591 | 0.425 |
| any | 189 / 59 | 0.509 [0.444, 0.573] | 0.528 | 0.610 | 0.417 | 0.488 | 0.538 |
