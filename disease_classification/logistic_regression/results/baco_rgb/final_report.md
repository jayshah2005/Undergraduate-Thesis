# Disease: logistic regression on Baco, RGB bands only

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression on the 3 training rows, C chosen on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.427 [0.380, 0.474] | — | 0.517 | 0.479 | 0.375 | 0.363 | 0.511 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.559 [0.507, 0.614] | — | 0.484 | 0.448 | 0.542 | 0.700 | 0.495 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.
