# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | +0.024 [-0.013, +0.058] | 0.624 | 0.676 | 0.494 | 0.566 | 0.599 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.505 | 0.693 | 0.569 | 0.528 | 0.669 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet (no max pool), mean pool | 0.480 | 0.497 | 0.463 |
| small ResNet (no max pool), attention pool | 0.524 | 0.475 | 0.496 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.561, 0.566, 0.594 | 0.574 ± 0.018 |
| small ResNet (no max pool), attention pool | 0.534, 0.557, 0.595 | 0.562 ± 0.031 |

Median best epoch: small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 11.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| small ResNet (no max pool), mean pool | 0.624 [0.599, 0.646] | +0.024 [-0.008, +0.054] | 0.551 | 0.637 | 0.651 | 0.608 | 0.641 |
| small ResNet (no max pool), attention pool | 0.590 [0.566, 0.614] | -0.009 [-0.042, +0.021] | 0.544 | 0.607 | 0.662 | 0.544 | 0.632 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| small ResNet (no max pool), mean pool | 0.530 | 0.565 | 0.530 |
| small ResNet (no max pool), attention pool | 0.451 | 0.527 | 0.537 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet (no max pool), mean pool | 0.618, 0.604, 0.598 | 0.606 ± 0.010 |
| small ResNet (no max pool), attention pool | 0.618, 0.587, 0.593 | 0.600 ± 0.016 |

Median best epoch: small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 4.
