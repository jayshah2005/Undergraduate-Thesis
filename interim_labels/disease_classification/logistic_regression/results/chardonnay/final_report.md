# Disease: logistic regression on Chardonnay, full spectrum

Leaf mean spectrum → SNV → logistic regression, trained on the 3 training rows (the validation row is used only by the CNN). Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | — | 0.483 | 0.511 | 0.556 | 0.325 | 0.572 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | — | 0.529 | 0.525 | 0.686 | 0.620 | 0.525 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

## Both diseases (50 + / 200 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.522 [0.458, 0.593] | — | 0.359 | 0.517 | 0.594 | 0.424 | 0.446 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

## Any disease vs healthy (160 + / 90 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.562 [0.499, 0.626] | — | 0.591 | 0.471 | 0.632 | 0.540 | 0.554 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.
