# Disease: logistic regression on Chardonnay, full spectrum

Leaf average of its per-pixel SNV spectra (RGB: per-pixel normalized rgb) → class-balanced logistic regression on the 3 training rows, C chosen on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. Every metric: `metrics.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.533 [0.473, 0.593] | — | 0.481 | 0.511 | 0.653 | 0.375 | 0.624 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.537, 0.656] | — | 0.494 | 0.450 | 0.790 | 0.507 | 0.700 |

Fold k tests on row Rk of Chardonnay; the next row (R1 after R5) is validation and the other 3 train.
