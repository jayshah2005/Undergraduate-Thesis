# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | +0.052 [+0.022, +0.083] | 0.577 | 0.615 | 0.545 | 0.581 | 0.527 |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | +0.022 [-0.002, +0.046] | 0.568 | 0.607 | 0.544 | 0.584 | 0.533 |
| small ResNet, mean ∥ max pool | 0.577 [0.553, 0.601] | +0.035 [-0.000, +0.069] | 0.537 | 0.646 | 0.535 | 0.586 | 0.524 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| small ResNet, mean pool | 0.524 | 0.525 | 0.446 |
| small ResNet, attention pool | 0.477 | 0.440 | 0.488 |
| small ResNet, mean ∥ max pool | 0.509 | 0.486 | 0.486 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.594, 0.547, 0.567 | 0.569 ± 0.023 |
| small ResNet, attention pool | 0.572, 0.546, 0.567 | 0.561 ± 0.014 |
| small ResNet, mean ∥ max pool | 0.536, 0.582, 0.551 | 0.556 ± 0.024 |

Median best epoch: small ResNet, mean pool 7, small ResNet, attention pool 4, small ResNet, mean ∥ max pool 7.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.018 [-0.045, +0.008] |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| small ResNet, mean pool | 0.659 [0.641, 0.678] | +0.015 [-0.005, +0.035] | 0.656 | 0.658 | 0.635 | 0.688 | 0.681 |
| small ResNet, attention pool | 0.668 [0.646, 0.688] | +0.024 [+0.003, +0.043] | 0.640 | 0.642 | 0.699 | 0.681 | 0.687 |
| small ResNet, mean ∥ max pool | 0.666 [0.647, 0.685] | +0.022 [+0.000, +0.044] | 0.639 | 0.675 | 0.677 | 0.652 | 0.703 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| small ResNet, mean pool | 0.497 | 0.502 | 0.531 | 0.492 |
| small ResNet, attention pool | 0.508 | 0.485 | 0.542 | 0.521 |
| small ResNet, mean ∥ max pool | 0.503 | 0.497 | 0.517 | 0.521 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.678, 0.665, 0.649 | 0.664 ± 0.015 |
| small ResNet, attention pool | 0.669, 0.653, 0.659 | 0.660 ± 0.008 |
| small ResNet, mean ∥ max pool | 0.659, 0.664, 0.652 | 0.658 ± 0.006 |

Median best epoch: small ResNet, mean pool 19, small ResNet, attention pool 17, small ResNet, mean ∥ max pool 15.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.007 [-0.008, +0.022] |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | — | 0.576 | 0.542 | 0.655 | 0.591 | 0.577 |
| small ResNet, mean pool | 0.597 [0.574, 0.620] | +0.014 [-0.014, +0.041] | 0.484 | 0.567 | 0.590 | 0.623 | 0.618 |
| small ResNet, attention pool | 0.598 [0.577, 0.619] | +0.015 [-0.012, +0.041] | 0.530 | 0.561 | 0.591 | 0.630 | 0.633 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.440 | 0.497 | 0.542 | 0.532 |
| small ResNet, mean pool | 0.508 | 0.491 | 0.497 | 0.507 |
| small ResNet, attention pool | 0.492 | 0.470 | 0.505 | 0.521 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.604, 0.597, 0.598 | 0.600 ± 0.003 |
| small ResNet, attention pool | 0.589, 0.595, 0.611 | 0.598 ± 0.011 |

Median best epoch: small ResNet, mean pool 11, small ResNet, attention pool 15.
