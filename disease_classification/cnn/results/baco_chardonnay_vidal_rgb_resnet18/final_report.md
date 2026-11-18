# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | +0.044 [+0.007, +0.080] | 0.557 | 0.628 | 0.540 | 0.509 | 0.571 |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | +0.068 [+0.033, +0.105] | 0.595 | 0.633 | 0.561 | 0.493 | 0.637 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| ResNet-18, mean pool | 0.523 | 0.509 | 0.459 |
| ResNet-18, attention pool | 0.549 | 0.471 | 0.520 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.591, 0.599, 0.570 | 0.587 ± 0.015 |
| ResNet-18, attention pool | 0.597, 0.592, 0.583 | 0.591 ± 0.007 |

Median best epoch: ResNet-18, mean pool 15, ResNet-18, attention pool 10.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| ResNet-18, mean pool | 0.605 [0.578, 0.632] | -0.004 [-0.036, +0.028] | 0.551 | 0.626 | 0.693 | 0.545 | 0.618 |
| ResNet-18, attention pool | 0.581 [0.557, 0.605] | -0.029 [-0.057, -0.000] | 0.559 | 0.534 | 0.616 | 0.522 | 0.663 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| ResNet-18, mean pool | 0.511 | 0.546 | 0.550 |
| ResNet-18, attention pool | 0.487 | 0.528 | 0.532 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.565, 0.615, 0.566 | 0.582 ± 0.029 |
| ResNet-18, attention pool | 0.580, 0.553, 0.582 | 0.572 ± 0.016 |

Median best epoch: ResNet-18, mean pool 18, ResNet-18, attention pool 11.
