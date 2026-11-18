# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | +0.035 [-0.008, +0.075] | 0.541 | 0.652 | 0.623 | 0.561 | 0.598 |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | +0.002 [-0.042, +0.043] | 0.567 | 0.629 | 0.571 | 0.515 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| ResNet-18, mean pool | 0.548 | 0.502 | 0.493 |
| ResNet-18, attention pool | 0.507 | 0.419 | 0.514 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.594, 0.609, 0.539 | 0.581 ± 0.037 |
| ResNet-18, attention pool | 0.540, 0.589, 0.570 | 0.566 ± 0.025 |

Median best epoch: ResNet-18, mean pool 7, ResNet-18, attention pool 11.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| ResNet-18, mean pool | 0.602 [0.580, 0.623] | +0.002 [-0.031, +0.032] | 0.570 | 0.589 | 0.621 | 0.572 | 0.641 |
| ResNet-18, attention pool | 0.591 [0.567, 0.615] | -0.009 [-0.043, +0.025] | 0.546 | 0.603 | 0.609 | 0.579 | 0.633 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| ResNet-18, mean pool | 0.501 | 0.535 | 0.522 |
| ResNet-18, attention pool | 0.489 | 0.542 | 0.531 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.569, 0.616, 0.574 | 0.586 ± 0.026 |
| ResNet-18, attention pool | 0.583, 0.547, 0.577 | 0.569 ± 0.019 |

Median best epoch: ResNet-18, mean pool 12, ResNet-18, attention pool 18.
