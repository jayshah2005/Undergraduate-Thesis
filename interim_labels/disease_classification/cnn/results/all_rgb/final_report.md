# Disease: whole-leaf CNN on all varieties, RGB bands only

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

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | — | 0.624 | 0.625 | 0.721 | 0.635 | 0.634 |
| CNN, mean pool | 0.650 [0.631, 0.668] | +0.009 [-0.012, +0.030] | 0.636 | 0.635 | 0.651 | 0.679 | 0.657 |
| CNN, attention pool | 0.667 [0.647, 0.686] | +0.026 [+0.003, +0.048] | 0.659 | 0.647 | 0.666 | 0.688 | 0.674 |
| small ResNet, mean pool | 0.662 [0.643, 0.680] | +0.021 [-0.001, +0.043] | 0.653 | 0.657 | 0.644 | 0.679 | 0.689 |
| small ResNet, attention pool | 0.662 [0.643, 0.680] | +0.022 [+0.001, +0.041] | 0.657 | 0.659 | 0.648 | 0.667 | 0.696 |
| small ResNet (no max pool), mean pool | 0.665 [0.647, 0.684] | +0.024 [+0.004, +0.045] | 0.671 | 0.653 | 0.659 | 0.688 | 0.668 |
| small ResNet (no max pool), attention pool | 0.656 [0.638, 0.674] | +0.015 [-0.006, +0.035] | 0.641 | 0.669 | 0.654 | 0.671 | 0.653 |
| ResNet-18, mean pool | 0.659 [0.639, 0.678] | +0.018 [-0.006, +0.041] | 0.646 | 0.633 | 0.676 | 0.658 | 0.701 |
| ResNet-18, attention pool | 0.663 [0.644, 0.683] | +0.022 [-0.000, +0.046] | 0.634 | 0.659 | 0.639 | 0.681 | 0.706 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.458 | 0.499 | 0.567 | 0.542 |
| CNN, mean pool | 0.497 | 0.489 | 0.517 | 0.471 |
| CNN, attention pool | 0.503 | 0.502 | 0.534 | 0.503 |
| small ResNet, mean pool | 0.489 | 0.493 | 0.511 | 0.535 |
| small ResNet, attention pool | 0.512 | 0.491 | 0.531 | 0.506 |
| small ResNet (no max pool), mean pool | 0.508 | 0.497 | 0.556 | 0.505 |
| small ResNet (no max pool), attention pool | 0.494 | 0.487 | 0.519 | 0.505 |
| ResNet-18, mean pool | 0.521 | 0.504 | 0.559 | 0.489 |
| ResNet-18, attention pool | 0.491 | 0.551 | 0.535 | 0.529 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.677, 0.655, 0.656 | 0.663 ± 0.012 |
| CNN, attention pool | 0.658, 0.672, 0.668 | 0.666 ± 0.007 |
| small ResNet, mean pool | 0.649, 0.658, 0.651 | 0.652 ± 0.005 |
| small ResNet, attention pool | 0.669, 0.655, 0.652 | 0.658 ± 0.009 |
| small ResNet (no max pool), mean pool | 0.662, 0.657, 0.660 | 0.660 ± 0.003 |
| small ResNet (no max pool), attention pool | 0.653, 0.649, 0.650 | 0.651 ± 0.003 |
| ResNet-18, mean pool | 0.656, 0.634, 0.623 | 0.638 ± 0.017 |
| ResNet-18, attention pool | 0.650, 0.645, 0.668 | 0.654 ± 0.012 |

Median best epoch: CNN, mean pool 15, CNN, attention pool 11, small ResNet, mean pool 15, small ResNet, attention pool 19, small ResNet (no max pool), mean pool 18, small ResNet (no max pool), attention pool 19, ResNet-18, mean pool 19, ResNet-18, attention pool 17.

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | — | 0.581 | 0.523 | 0.673 | 0.598 | 0.567 |
| CNN, mean pool | 0.614 [0.592, 0.635] | +0.030 [+0.001, +0.056] | 0.527 | 0.567 | 0.606 | 0.645 | 0.651 |
| CNN, attention pool | 0.597 [0.575, 0.619] | +0.013 [-0.016, +0.041] | 0.491 | 0.543 | 0.606 | 0.627 | 0.618 |
| small ResNet, mean pool | 0.593 [0.572, 0.616] | +0.009 [-0.019, +0.036] | 0.493 | 0.548 | 0.609 | 0.638 | 0.591 |
| small ResNet, attention pool | 0.597 [0.577, 0.619] | +0.013 [-0.015, +0.042] | 0.559 | 0.543 | 0.596 | 0.615 | 0.608 |
| small ResNet (no max pool), mean pool | 0.577 [0.554, 0.598] | -0.007 [-0.034, +0.018] | 0.450 | 0.537 | 0.593 | 0.612 | 0.604 |
| small ResNet (no max pool), attention pool | 0.599 [0.577, 0.621] | +0.016 [-0.011, +0.040] | 0.564 | 0.553 | 0.603 | 0.638 | 0.612 |
| ResNet-18, mean pool | 0.621 [0.597, 0.644] | +0.037 [+0.005, +0.067] | 0.475 | 0.570 | 0.631 | 0.611 | 0.627 |
| ResNet-18, attention pool | 0.604 [0.580, 0.627] | +0.020 [-0.011, +0.049] | 0.539 | 0.530 | 0.632 | 0.600 | 0.581 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.449 | 0.495 | 0.542 | 0.541 |
| CNN, mean pool | 0.487 | 0.563 | 0.500 | 0.536 |
| CNN, attention pool | 0.521 | 0.538 | 0.500 | 0.493 |
| small ResNet, mean pool | 0.497 | 0.541 | 0.497 | 0.499 |
| small ResNet, attention pool | 0.523 | 0.468 | 0.499 | 0.498 |
| small ResNet (no max pool), mean pool | 0.484 | 0.407 | 0.487 | 0.505 |
| small ResNet (no max pool), attention pool | 0.493 | 0.530 | 0.518 | 0.510 |
| ResNet-18, mean pool | 0.518 | 0.555 | 0.579 | 0.547 |
| ResNet-18, attention pool | 0.548 | 0.598 | 0.542 | 0.527 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.616, 0.604, 0.613 | 0.611 ± 0.006 |
| CNN, attention pool | 0.599, 0.602, 0.599 | 0.600 ± 0.002 |
| small ResNet, mean pool | 0.599, 0.588, 0.603 | 0.597 ± 0.008 |
| small ResNet, attention pool | 0.591, 0.616, 0.594 | 0.600 ± 0.013 |
| small ResNet (no max pool), mean pool | 0.582, 0.581, 0.588 | 0.584 ± 0.004 |
| small ResNet (no max pool), attention pool | 0.594, 0.584, 0.594 | 0.591 ± 0.006 |
| ResNet-18, mean pool | 0.621, 0.612, 0.610 | 0.614 ± 0.006 |
| ResNet-18, attention pool | 0.593, 0.592, 0.571 | 0.585 ± 0.012 |

Median best epoch: CNN, mean pool 21, CNN, attention pool 11, small ResNet, mean pool 11, small ResNet, attention pool 18, small ResNet (no max pool), mean pool 16, small ResNet (no max pool), attention pool 11, ResNet-18, mean pool 14, ResNet-18, attention pool 11.
