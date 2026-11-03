# Disease logistic regression: Chardonnay, full spectrum

Each row is tested once; the model trains on 3 other rows (the next row, the CNN's validation row, stays unused). Balanced accuracy over all test rows (0.5 = chance), P ≥ 0.5 counts as positive, with a 95% bootstrap interval.

| task | leaves (+ / −) | bal. acc. [95% CI] | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| leafroll | 122 / 128 | 0.537 [0.477, 0.596] | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |
| red_blotch | 88 / 162 | 0.605 [0.547, 0.662] | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |
| both | 50 / 200 | 0.522 [0.458, 0.593] | 0.359 | 0.517 | 0.594 | 0.424 | 0.446 |
| any | 160 / 90 | 0.562 [0.499, 0.626] | 0.591 | 0.471 | 0.632 | 0.540 | 0.554 |
