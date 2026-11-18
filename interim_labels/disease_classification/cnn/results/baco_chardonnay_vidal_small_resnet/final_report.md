# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | +0.014 [-0.022, +0.048] | 0.552 | 0.623 | 0.550 | 0.518 | 0.676 |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | +0.045 [+0.006, +0.082] | 0.587 | 0.703 | 0.616 | 0.541 | 0.613 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| small ResNet, mean pool | 0.510 | 0.445 | 0.509 |
| small ResNet, attention pool | 0.507 | 0.489 | 0.513 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.597, 0.599, 0.557 | 0.584 ± 0.024 |
| small ResNet, attention pool | 0.578, 0.598, 0.605 | 0.593 ± 0.014 |

Median best epoch: small ResNet, mean pool 16, small ResNet, attention pool 18.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | — | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| small ResNet, mean pool | 0.624 [0.605, 0.644] | +0.028 [-0.002, +0.056] | 0.572 | 0.572 | 0.706 | 0.607 | 0.636 |
| small ResNet, attention pool | 0.604 [0.581, 0.627] | +0.007 [-0.025, +0.037] | 0.551 | 0.579 | 0.691 | 0.595 | 0.632 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.460 | 0.548 | 0.546 |
| small ResNet, mean pool | 0.499 | 0.563 | 0.567 |
| small ResNet, attention pool | 0.476 | 0.549 | 0.559 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.580, 0.627, 0.590 | 0.599 ± 0.025 |
| small ResNet, attention pool | 0.594, 0.576, 0.620 | 0.597 ± 0.022 |

Median best epoch: small ResNet, mean pool 16, small ResNet, attention pool 11.
