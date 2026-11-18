# Disease: whole-leaf CNN on all varieties, RGB bands only

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

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| small ResNet (no max pool), mean pool | 0.653 [0.634, 0.672] | +0.009 [-0.010, +0.028] | 0.649 | 0.668 | 0.637 | 0.681 | 0.650 |
| small ResNet (no max pool), attention pool | 0.659 [0.641, 0.678] | +0.015 [-0.004, +0.033] | 0.645 | 0.666 | 0.670 | 0.692 | 0.650 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| small ResNet (no max pool), mean pool | 0.494 | 0.487 | 0.539 | 0.487 |
| small ResNet (no max pool), attention pool | 0.509 | 0.479 | 0.536 | 0.512 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.659, 0.660, 0.646 | 0.655 ± 0.008 |
| small ResNet (no max pool), attention pool | 0.652, 0.662, 0.649 | 0.654 ± 0.007 |

Median best epoch: small ResNet (no max pool), mean pool 14, small ResNet (no max pool), attention pool 15.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | — | 0.576 | 0.542 | 0.655 | 0.591 | 0.577 |
| small ResNet (no max pool), mean pool | 0.587 [0.564, 0.611] | +0.005 [-0.019, +0.029] | 0.563 | 0.559 | 0.577 | 0.615 | 0.591 |
| small ResNet (no max pool), attention pool | 0.591 [0.572, 0.613] | +0.009 [-0.015, +0.032] | 0.531 | 0.543 | 0.595 | 0.619 | 0.619 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.440 | 0.497 | 0.542 | 0.532 |
| small ResNet (no max pool), mean pool | 0.482 | 0.409 | 0.518 | 0.497 |
| small ResNet (no max pool), attention pool | 0.476 | 0.464 | 0.518 | 0.511 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.602, 0.592, 0.601 | 0.598 ± 0.006 |
| small ResNet (no max pool), attention pool | 0.585, 0.576, 0.601 | 0.588 ± 0.013 |

Median best epoch: small ResNet (no max pool), mean pool 17, small ResNet (no max pool), attention pool 19.
