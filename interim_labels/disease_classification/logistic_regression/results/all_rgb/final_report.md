# Disease: logistic regression on all varieties, RGB bands only

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Left out (fewer than 5 positive or negative leaves): Cab Franc.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | — | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.458 | 0.499 | 0.567 | 0.542 |

## Both diseases (115 + / 633 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.492 [0.456, 0.529] | — | 0.503 | 0.527 | 0.470 | 0.511 | 0.407 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Left out (fewer than 5 positive or negative leaves): Cab Franc.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.405 | 0.553 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | — | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.449 | 0.495 | 0.542 | 0.541 |
