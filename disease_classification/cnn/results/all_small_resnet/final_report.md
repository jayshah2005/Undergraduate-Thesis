# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | +0.014 [-0.022, +0.049] | 0.572 | 0.643 | 0.548 | 0.540 | 0.634 |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | +0.040 [+0.002, +0.075] | 0.489 | 0.668 | 0.611 | 0.522 | 0.645 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet, mean pool | 0.503 | 0.442 | 0.505 |
| small ResNet, attention pool | 0.518 | 0.520 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.584, 0.575, 0.559 | 0.573 ± 0.012 |
| small ResNet, attention pool | 0.617, 0.609, 0.601 | 0.609 ± 0.008 |

Median best epoch: small ResNet, mean pool 17, small ResNet, attention pool 18.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | +0.036 [+0.008, +0.063] | 0.620 | 0.655 | 0.731 | 0.690 | 0.691 |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | +0.040 [+0.014, +0.065] | 0.652 | 0.664 | 0.748 | 0.669 | 0.695 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| small ResNet, mean pool | 0.504 | 0.512 | 0.555 | 0.537 |
| small ResNet, attention pool | 0.474 | 0.524 | 0.538 | 0.541 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.684, 0.671, 0.675 | 0.677 ± 0.007 |
| small ResNet, attention pool | 0.689, 0.673, 0.656 | 0.672 ± 0.016 |

Median best epoch: small ResNet, mean pool 11, small ResNet, attention pool 11.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | — | 0.427 | 0.625 | 0.628 | 0.536 | 0.594 |
| small ResNet, mean pool | 0.637 [0.613, 0.658] | +0.081 [+0.042, +0.122] | 0.588 | 0.605 | 0.622 | 0.611 | 0.654 |
| small ResNet, attention pool | 0.627 [0.601, 0.650] | +0.070 [+0.033, +0.111] | 0.590 | 0.554 | 0.636 | 0.585 | 0.590 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.448 | 0.444 | 0.481 | 0.590 |
| small ResNet, mean pool | 0.513 | 0.557 | 0.563 | 0.569 |
| small ResNet, attention pool | 0.593 | 0.586 | 0.553 | 0.539 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.608, 0.620, 0.604 | 0.611 ± 0.008 |
| small ResNet, attention pool | 0.611, 0.598, 0.633 | 0.614 ± 0.017 |

Median best epoch: small ResNet, mean pool 7, small ResNet, attention pool 6.
