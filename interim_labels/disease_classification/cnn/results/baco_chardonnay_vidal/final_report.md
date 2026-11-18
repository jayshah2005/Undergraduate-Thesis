# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | — | 0.558 | 0.560 | 0.701 | 0.504 | 0.601 |
| CNN, mean pool | 0.604 [0.582, 0.626] | +0.028 [-0.009, +0.061] | 0.640 | 0.662 | 0.490 | 0.527 | 0.649 |
| CNN, attention pool | 0.620 [0.598, 0.642] | +0.043 [+0.006, +0.077] | 0.551 | 0.636 | 0.607 | 0.513 | 0.686 |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | +0.014 [-0.022, +0.048] | 0.552 | 0.623 | 0.550 | 0.518 | 0.676 |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | +0.045 [+0.006, +0.082] | 0.587 | 0.703 | 0.616 | 0.541 | 0.613 |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | +0.021 [-0.016, +0.055] | 0.572 | 0.662 | 0.495 | 0.513 | 0.688 |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | +0.030 [-0.009, +0.065] | 0.504 | 0.684 | 0.571 | 0.578 | 0.625 |
| ResNet-18, mean pool | 0.601 [0.572, 0.627] | +0.024 [-0.020, +0.065] | 0.620 | 0.640 | 0.626 | 0.479 | 0.552 |
| ResNet-18, attention pool | 0.577 [0.549, 0.603] | +0.000 [-0.042, +0.041] | 0.563 | 0.630 | 0.574 | 0.533 | 0.559 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.470 | 0.482 | 0.527 |
| CNN, mean pool | 0.519 | 0.476 | 0.527 |
| CNN, attention pool | 0.525 | 0.521 | 0.531 |
| small ResNet, mean pool | 0.510 | 0.445 | 0.509 |
| small ResNet, attention pool | 0.507 | 0.489 | 0.513 |
| small ResNet (no max pool), mean pool | 0.528 | 0.458 | 0.498 |
| small ResNet (no max pool), attention pool | 0.538 | 0.467 | 0.544 |
| ResNet-18, mean pool | 0.507 | 0.494 | 0.533 |
| ResNet-18, attention pool | 0.522 | 0.443 | 0.476 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.611, 0.630, 0.591 | 0.611 ± 0.019 |
| CNN, attention pool | 0.612, 0.607, 0.611 | 0.610 ± 0.002 |
| small ResNet, mean pool | 0.597, 0.599, 0.557 | 0.584 ± 0.024 |
| small ResNet, attention pool | 0.578, 0.598, 0.605 | 0.593 ± 0.014 |
| small ResNet (no max pool), mean pool | 0.597, 0.556, 0.589 | 0.581 ± 0.021 |
| small ResNet (no max pool), attention pool | 0.577, 0.573, 0.594 | 0.582 ± 0.011 |
| ResNet-18, mean pool | 0.593, 0.605, 0.554 | 0.584 ± 0.027 |
| ResNet-18, attention pool | 0.541, 0.585, 0.549 | 0.558 ± 0.023 |

Median best epoch: CNN, mean pool 20, CNN, attention pool 16, small ResNet, mean pool 16, small ResNet, attention pool 18, small ResNet (no max pool), mean pool 12, small ResNet (no max pool), attention pool 15, ResNet-18, mean pool 9, ResNet-18, attention pool 12.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | — | 0.549 | 0.606 | 0.686 | 0.528 | 0.603 |
| CNN, mean pool | 0.608 [0.585, 0.631] | +0.012 [-0.019, +0.043] | 0.598 | 0.589 | 0.609 | 0.569 | 0.653 |
| CNN, attention pool | 0.605 [0.583, 0.626] | +0.008 [-0.022, +0.037] | 0.559 | 0.579 | 0.681 | 0.571 | 0.632 |
| small ResNet, mean pool | 0.624 [0.605, 0.644] | +0.028 [-0.002, +0.056] | 0.572 | 0.572 | 0.706 | 0.607 | 0.636 |
| small ResNet, attention pool | 0.604 [0.581, 0.627] | +0.007 [-0.025, +0.037] | 0.551 | 0.579 | 0.691 | 0.595 | 0.632 |
| small ResNet (no max pool), mean pool | 0.616 [0.592, 0.639] | +0.019 [-0.011, +0.050] | 0.575 | 0.613 | 0.617 | 0.594 | 0.657 |
| small ResNet (no max pool), attention pool | 0.610 [0.589, 0.630] | +0.013 [-0.017, +0.041] | 0.572 | 0.579 | 0.681 | 0.576 | 0.639 |
| ResNet-18, mean pool | 0.590 [0.567, 0.614] | -0.007 [-0.041, +0.026] | 0.564 | 0.580 | 0.652 | 0.571 | 0.624 |
| ResNet-18, attention pool | 0.577 [0.552, 0.601] | -0.020 [-0.054, +0.014] | 0.512 | 0.593 | 0.664 | 0.567 | 0.580 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.460 | 0.548 | 0.546 |
| CNN, mean pool | 0.483 | 0.565 | 0.559 |
| CNN, attention pool | 0.452 | 0.577 | 0.562 |
| small ResNet, mean pool | 0.499 | 0.563 | 0.567 |
| small ResNet, attention pool | 0.476 | 0.549 | 0.559 |
| small ResNet (no max pool), mean pool | 0.538 | 0.551 | 0.525 |
| small ResNet (no max pool), attention pool | 0.474 | 0.538 | 0.571 |
| ResNet-18, mean pool | 0.493 | 0.533 | 0.539 |
| ResNet-18, attention pool | 0.469 | 0.526 | 0.540 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.593, 0.612, 0.596 | 0.600 ± 0.010 |
| CNN, attention pool | 0.586, 0.571, 0.615 | 0.590 ± 0.023 |
| small ResNet, mean pool | 0.580, 0.627, 0.590 | 0.599 ± 0.025 |
| small ResNet, attention pool | 0.594, 0.576, 0.620 | 0.597 ± 0.022 |
| small ResNet (no max pool), mean pool | 0.618, 0.584, 0.596 | 0.599 ± 0.017 |
| small ResNet (no max pool), attention pool | 0.581, 0.595, 0.596 | 0.591 ± 0.008 |
| ResNet-18, mean pool | 0.585, 0.590, 0.556 | 0.577 ± 0.018 |
| ResNet-18, attention pool | 0.573, 0.557, 0.553 | 0.561 ± 0.010 |

Median best epoch: CNN, mean pool 12, CNN, attention pool 6, small ResNet, mean pool 16, small ResNet, attention pool 11, small ResNet (no max pool), mean pool 5, small ResNet (no max pool), attention pool 7, ResNet-18, mean pool 12, ResNet-18, attention pool 14.
