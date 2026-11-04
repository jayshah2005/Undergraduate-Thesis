# Disease: logistic regression on Baco, RGB bands only

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.481 [0.448, 0.512] | — | 0.507 | 0.446 | 0.526 | 0.485 | 0.481 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | — | 0.517 | 0.412 | 0.504 | 0.350 | 0.525 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | — | 0.497 | 0.481 | 0.549 | 0.620 | 0.436 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Both diseases (53 + / 195 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.523 [0.455, 0.590] | — | 0.562 | 0.478 | 0.445 | 0.568 | 0.600 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Any disease vs healthy (189 + / 59 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.606 [0.543, 0.670] | — | 0.635 | 0.470 | 0.444 | 0.659 | 0.481 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.
