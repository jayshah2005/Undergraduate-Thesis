# Disease: logistic regression on Vidal, RGB bands only

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.477 [0.434, 0.520] | — | 0.531 | 0.473 | 0.369 | 0.472 | 0.591 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Both diseases (12 + / 238 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.486 [0.369, 0.608] | — | 0.353 | 0.615 | 0.296 | 0.533 | 0.714 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Any disease vs healthy (182 + / 68 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.480, 0.611] | — | 0.613 | 0.566 | 0.515 | 0.505 | 0.531 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.
