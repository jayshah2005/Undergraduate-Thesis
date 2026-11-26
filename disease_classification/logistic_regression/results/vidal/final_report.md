# Disease: logistic regression on Vidal, full spectrum

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression on the 3 training rows, C chosen on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.457 [0.370, 0.543] | — | 0.412 | 0.530 | 0.255 | 0.589 | 0.458 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.476, 0.601] | — | 0.581 | 0.616 | 0.589 | 0.494 | 0.506 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.
