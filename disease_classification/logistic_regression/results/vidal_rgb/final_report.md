# Disease: logistic regression on Vidal, RGB bands only

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression on the 3 training rows, C chosen on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.422 [0.336, 0.510] | — | 0.424 | 0.477 | 0.223 | 0.435 | 0.688 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.528 [0.461, 0.589] | — | 0.515 | 0.585 | 0.583 | 0.493 | 0.478 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.
