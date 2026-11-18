# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| CNN, mean pool | 0.565 [0.535, 0.593] | +0.021 [-0.004, +0.045] | 0.557 | 0.627 | 0.546 | 0.546 | 0.578 |
| CNN, attention pool | 0.580 [0.552, 0.607] | +0.036 [+0.010, +0.064] | 0.583 | 0.627 | 0.568 | 0.559 | 0.574 |
| small ResNet, mean pool | 0.585 [0.556, 0.614] | +0.041 [+0.015, +0.069] | 0.579 | 0.615 | 0.549 | 0.586 | 0.552 |
| small ResNet, attention pool | 0.564 [0.535, 0.590] | +0.020 [-0.006, +0.044] | 0.583 | 0.607 | 0.549 | 0.543 | 0.552 |
| small ResNet (no max pool), mean pool | 0.555 [0.525, 0.585] | +0.011 [-0.014, +0.035] | 0.569 | 0.603 | 0.524 | 0.551 | 0.542 |
| small ResNet (no max pool), attention pool | 0.574 [0.546, 0.601] | +0.030 [+0.003, +0.055] | 0.586 | 0.613 | 0.522 | 0.551 | 0.584 |
| ResNet-18, mean pool | 0.577 [0.551, 0.600] | +0.033 [-0.003, +0.068] | 0.580 | 0.628 | 0.543 | 0.549 | 0.524 |
| ResNet-18, attention pool | 0.596 [0.568, 0.623] | +0.052 [+0.016, +0.091] | 0.605 | 0.633 | 0.565 | 0.500 | 0.567 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| CNN, mean pool | 0.462 | 0.436 | 0.493 |
| CNN, attention pool | 0.540 | 0.423 | 0.502 |
| small ResNet, mean pool | 0.495 | 0.474 | 0.493 |
| small ResNet, attention pool | 0.483 | 0.412 | 0.496 |
| small ResNet (no max pool), mean pool | 0.468 | 0.427 | 0.487 |
| small ResNet (no max pool), attention pool | 0.499 | 0.423 | 0.502 |
| ResNet-18, mean pool | 0.498 | 0.463 | 0.489 |
| ResNet-18, attention pool | 0.530 | 0.463 | 0.491 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.578, 0.548, 0.556 | 0.561 ± 0.015 |
| CNN, attention pool | 0.569, 0.572, 0.589 | 0.576 ± 0.011 |
| small ResNet, mean pool | 0.598, 0.544, 0.556 | 0.566 ± 0.029 |
| small ResNet, attention pool | 0.560, 0.554, 0.566 | 0.560 ± 0.006 |
| small ResNet (no max pool), mean pool | 0.570, 0.508, 0.571 | 0.549 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.569, 0.558, 0.567 | 0.565 ± 0.006 |
| ResNet-18, mean pool | 0.569, 0.576, 0.579 | 0.574 ± 0.005 |
| ResNet-18, attention pool | 0.578, 0.582, 0.582 | 0.581 ± 0.002 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 2, small ResNet, mean pool 7, small ResNet, attention pool 7, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 7, ResNet-18, mean pool 11, ResNet-18, attention pool 10.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | — | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| CNN, mean pool | 0.602 [0.576, 0.628] | -0.003 [-0.021, +0.018] | 0.612 | 0.567 | 0.673 | 0.550 | 0.641 |
| CNN, attention pool | 0.607 [0.582, 0.633] | +0.003 [-0.015, +0.022] | 0.612 | 0.553 | 0.649 | 0.597 | 0.662 |
| small ResNet, mean pool | 0.608 [0.583, 0.633] | +0.003 [-0.016, +0.023] | 0.622 | 0.568 | 0.648 | 0.582 | 0.657 |
| small ResNet, attention pool | 0.598 [0.572, 0.625] | -0.006 [-0.027, +0.013] | 0.592 | 0.567 | 0.648 | 0.584 | 0.635 |
| small ResNet (no max pool), mean pool | 0.605 [0.579, 0.630] | +0.001 [-0.018, +0.021] | 0.609 | 0.544 | 0.675 | 0.583 | 0.650 |
| small ResNet (no max pool), attention pool | 0.608 [0.582, 0.633] | +0.004 [-0.015, +0.023] | 0.597 | 0.547 | 0.686 | 0.583 | 0.649 |
| ResNet-18, mean pool | 0.562 [0.536, 0.589] | -0.042 [-0.074, -0.010] | 0.489 | 0.595 | 0.589 | 0.544 | 0.603 |
| ResNet-18, attention pool | 0.587 [0.562, 0.613] | -0.017 [-0.048, +0.011] | 0.541 | 0.621 | 0.679 | 0.539 | 0.562 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.434 | 0.610 | 0.551 |
| CNN, mean pool | 0.455 | 0.591 | 0.546 |
| CNN, attention pool | 0.483 | 0.565 | 0.562 |
| small ResNet, mean pool | 0.473 | 0.583 | 0.552 |
| small ResNet, attention pool | 0.464 | 0.567 | 0.543 |
| small ResNet (no max pool), mean pool | 0.477 | 0.576 | 0.549 |
| small ResNet (no max pool), attention pool | 0.467 | 0.594 | 0.546 |
| ResNet-18, mean pool | 0.493 | 0.494 | 0.522 |
| ResNet-18, attention pool | 0.527 | 0.514 | 0.542 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.587, 0.576, 0.596 | 0.587 ± 0.010 |
| CNN, attention pool | 0.596, 0.603, 0.589 | 0.596 ± 0.007 |
| small ResNet, mean pool | 0.590, 0.602, 0.607 | 0.600 ± 0.009 |
| small ResNet, attention pool | 0.583, 0.600, 0.601 | 0.595 ± 0.010 |
| small ResNet (no max pool), mean pool | 0.578, 0.603, 0.601 | 0.594 ± 0.014 |
| small ResNet (no max pool), attention pool | 0.595, 0.608, 0.605 | 0.603 ± 0.007 |
| ResNet-18, mean pool | 0.567, 0.531, 0.510 | 0.536 ± 0.029 |
| ResNet-18, attention pool | 0.559, 0.593, 0.582 | 0.578 ± 0.018 |

Median best epoch: CNN, mean pool 8, CNN, attention pool 10, small ResNet, mean pool 12, small ResNet, attention pool 6, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 5, ResNet-18, mean pool 13, ResNet-18, attention pool 13.
