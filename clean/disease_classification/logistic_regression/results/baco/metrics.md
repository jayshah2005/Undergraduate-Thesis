# Disease: logistic regression on Baco, full spectrum

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.515 [0.473, 0.558] | — | 0.508 | 0.556 | 0.486 | 0.561 | 0.475 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Both diseases (53 + / 195 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.470 [0.401, 0.536] | — | 0.413 | 0.559 | 0.418 | 0.591 | 0.425 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Any disease vs healthy (189 + / 59 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.509 [0.444, 0.573] | — | 0.528 | 0.610 | 0.417 | 0.488 | 0.538 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.
