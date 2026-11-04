# Disease: logistic regression on Vidal, full spectrum

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.548 [0.511, 0.585] | — | 0.532 | 0.548 | 0.516 | 0.527 | 0.493 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | — | 0.593 | 0.489 | 0.469 | 0.488 | 0.417 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | — | 0.471 | 0.608 | 0.562 | 0.566 | 0.570 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Both diseases (12 + / 238 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.479 [0.424, 0.562] | — | 0.571 | 0.292 | 0.490 | 0.500 | 0.459 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Any disease vs healthy (182 + / 68 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.516, 0.638] | — | 0.500 | 0.585 | 0.589 | 0.643 | 0.568 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.
