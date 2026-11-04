# Disease logistic regression: Vidal, RGB bands only

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 25 / 225 | 0.416 [0.338, 0.493] | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| red_blotch | 169 / 81 | 0.538 [0.474, 0.598] | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| both | 12 / 238 | 0.486 [0.369, 0.608] | 0.353 | 0.615 | 0.296 | 0.533 | 0.714 |
| any | 182 / 68 | 0.547 [0.480, 0.611] | 0.613 | 0.566 | 0.515 | 0.505 | 0.531 |
