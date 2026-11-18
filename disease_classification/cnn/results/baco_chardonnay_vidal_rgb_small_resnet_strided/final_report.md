# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| small ResNet (no max pool), mean pool | 0.557 [0.527, 0.589] | +0.015 [-0.010, +0.040] | 0.582 | 0.609 | 0.519 | 0.563 | 0.542 |
| small ResNet (no max pool), attention pool | 0.552 [0.521, 0.581] | +0.009 [-0.017, +0.034] | 0.577 | 0.613 | 0.517 | 0.552 | 0.529 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| small ResNet (no max pool), mean pool | 0.477 | 0.431 | 0.501 |
| small ResNet (no max pool), attention pool | 0.483 | 0.412 | 0.500 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.592, 0.520, 0.565 | 0.559 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.567, 0.525, 0.583 | 0.558 ± 0.030 |

Median best epoch: small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 6.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| small ResNet (no max pool), mean pool | 0.607 [0.581, 0.633] | -0.003 [-0.021, +0.016] | 0.586 | 0.575 | 0.671 | 0.592 | 0.653 |
| small ResNet (no max pool), attention pool | 0.615 [0.590, 0.639] | +0.005 [-0.011, +0.021] | 0.619 | 0.560 | 0.705 | 0.585 | 0.655 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| small ResNet (no max pool), mean pool | 0.461 | 0.594 | 0.545 |
| small ResNet (no max pool), attention pool | 0.471 | 0.602 | 0.536 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.593, 0.586, 0.592 | 0.590 ± 0.004 |
| small ResNet (no max pool), attention pool | 0.609, 0.609, 0.586 | 0.601 ± 0.013 |

Median best epoch: small ResNet (no max pool), mean pool 4, small ResNet (no max pool), attention pool 6.
