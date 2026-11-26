# Disease: logistic regression on Baco, full spectrum

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression on the 3 training rows, C chosen on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.438 [0.385, 0.493] | — | 0.463 | 0.490 | 0.432 | 0.375 | 0.536 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.548 [0.489, 0.601] | — | 0.521 | 0.490 | 0.523 | 0.620 | 0.524 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.
