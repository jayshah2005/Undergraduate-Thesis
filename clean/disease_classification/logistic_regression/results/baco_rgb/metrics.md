# Disease logistic regression: Baco, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 107 / 141 | 0.418 [0.371, 0.465] | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |
| red_blotch | 135 / 113 | 0.543 [0.494, 0.595] | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |
| both | 53 / 195 | 0.523 [0.455, 0.590] | 0.562 | 0.478 | 0.445 | 0.568 | 0.600 |
| any | 189 / 59 | 0.606 [0.543, 0.670] | 0.635 | 0.470 | 0.444 | 0.659 | 0.481 |
