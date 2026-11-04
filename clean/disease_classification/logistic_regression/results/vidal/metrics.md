# Disease logistic regression: Vidal, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 25 / 225 | 0.544 [0.491, 0.596] | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |
| red_blotch | 169 / 81 | 0.552 [0.494, 0.610] | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |
| both | 12 / 238 | 0.479 [0.424, 0.562] | 0.571 | 0.292 | 0.490 | 0.500 | 0.459 |
| any | 182 / 68 | 0.577 [0.516, 0.638] | 0.500 | 0.585 | 0.589 | 0.643 | 0.568 |
