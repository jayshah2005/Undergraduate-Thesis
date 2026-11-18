# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| CNN, mean pool | 0.586 [0.566, 0.606] | +0.014 [-0.021, +0.048] | 0.593 | 0.662 | 0.489 | 0.536 | 0.634 |
| CNN, attention pool | 0.597 [0.578, 0.618] | +0.026 [-0.010, +0.061] | 0.570 | 0.697 | 0.602 | 0.532 | 0.589 |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | +0.014 [-0.022, +0.049] | 0.572 | 0.643 | 0.548 | 0.540 | 0.634 |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | +0.040 [+0.002, +0.075] | 0.489 | 0.668 | 0.611 | 0.522 | 0.645 |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | +0.024 [-0.013, +0.058] | 0.624 | 0.676 | 0.494 | 0.566 | 0.599 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.505 | 0.693 | 0.569 | 0.528 | 0.669 |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | +0.035 [-0.008, +0.075] | 0.541 | 0.652 | 0.623 | 0.561 | 0.598 |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | +0.002 [-0.042, +0.043] | 0.567 | 0.629 | 0.571 | 0.515 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| CNN, mean pool | 0.494 | 0.457 | 0.518 |
| CNN, attention pool | 0.516 | 0.454 | 0.468 |
| small ResNet, mean pool | 0.503 | 0.442 | 0.505 |
| small ResNet, attention pool | 0.518 | 0.520 | 0.502 |
| small ResNet (no max pool), mean pool | 0.480 | 0.497 | 0.463 |
| small ResNet (no max pool), attention pool | 0.524 | 0.475 | 0.496 |
| ResNet-18, mean pool | 0.548 | 0.502 | 0.493 |
| ResNet-18, attention pool | 0.507 | 0.419 | 0.514 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.572, 0.628, 0.551 | 0.584 ± 0.040 |
| CNN, attention pool | 0.591, 0.537, 0.578 | 0.569 ± 0.029 |
| small ResNet, mean pool | 0.584, 0.575, 0.559 | 0.573 ± 0.012 |
| small ResNet, attention pool | 0.617, 0.609, 0.601 | 0.609 ± 0.008 |
| small ResNet (no max pool), mean pool | 0.561, 0.566, 0.594 | 0.574 ± 0.018 |
| small ResNet (no max pool), attention pool | 0.534, 0.557, 0.595 | 0.562 ± 0.031 |
| ResNet-18, mean pool | 0.594, 0.609, 0.539 | 0.581 ± 0.037 |
| ResNet-18, attention pool | 0.540, 0.589, 0.570 | 0.566 ± 0.025 |

Median best epoch: CNN, mean pool 18, CNN, attention pool 14, small ResNet, mean pool 17, small ResNet, attention pool 18, small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 11, ResNet-18, mean pool 7, ResNet-18, attention pool 11.

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| CNN, mean pool | 0.691 [0.672, 0.709] | +0.044 [+0.019, +0.070] | 0.689 | 0.690 | 0.709 | 0.673 | 0.704 |
| CNN, attention pool | 0.666 [0.646, 0.685] | +0.020 [-0.007, +0.047] | 0.605 | 0.655 | 0.680 | 0.673 | 0.692 |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | +0.036 [+0.008, +0.063] | 0.620 | 0.655 | 0.731 | 0.690 | 0.691 |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | +0.040 [+0.014, +0.065] | 0.652 | 0.664 | 0.748 | 0.669 | 0.695 |
| small ResNet (no max pool), mean pool | 0.683 [0.665, 0.700] | +0.036 [+0.012, +0.061] | 0.651 | 0.683 | 0.741 | 0.662 | 0.680 |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | +0.027 [-0.000, +0.053] | 0.677 | 0.657 | 0.689 | 0.675 | 0.686 |
| ResNet-18, mean pool | 0.672 [0.647, 0.693] | +0.025 [-0.004, +0.053] | 0.640 | 0.684 | 0.644 | 0.681 | 0.705 |
| ResNet-18, attention pool | 0.667 [0.643, 0.689] | +0.020 [-0.010, +0.049] | 0.627 | 0.671 | 0.658 | 0.646 | 0.726 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| CNN, mean pool | 0.499 | 0.522 | 0.555 | 0.539 |
| CNN, attention pool | 0.500 | 0.510 | 0.539 | 0.505 |
| small ResNet, mean pool | 0.504 | 0.512 | 0.555 | 0.537 |
| small ResNet, attention pool | 0.474 | 0.524 | 0.538 | 0.541 |
| small ResNet (no max pool), mean pool | 0.494 | 0.531 | 0.554 | 0.501 |
| small ResNet (no max pool), attention pool | 0.466 | 0.563 | 0.527 | 0.530 |
| ResNet-18, mean pool | 0.460 | 0.701 | 0.545 | 0.529 |
| ResNet-18, attention pool | 0.472 | 0.537 | 0.541 | 0.518 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.674, 0.664, 0.678 | 0.672 ± 0.007 |
| CNN, attention pool | 0.664, 0.658, 0.670 | 0.664 ± 0.006 |
| small ResNet, mean pool | 0.684, 0.671, 0.675 | 0.677 ± 0.007 |
| small ResNet, attention pool | 0.689, 0.673, 0.656 | 0.672 ± 0.016 |
| small ResNet (no max pool), mean pool | 0.669, 0.654, 0.684 | 0.669 ± 0.015 |
| small ResNet (no max pool), attention pool | 0.651, 0.645, 0.658 | 0.651 ± 0.007 |
| ResNet-18, mean pool | 0.654, 0.635, 0.672 | 0.654 ± 0.018 |
| ResNet-18, attention pool | 0.667, 0.679, 0.643 | 0.663 ± 0.018 |

Median best epoch: CNN, mean pool 10, CNN, attention pool 9, small ResNet, mean pool 11, small ResNet, attention pool 11, small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 9, ResNet-18, mean pool 20, ResNet-18, attention pool 18.

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | — | 0.427 | 0.625 | 0.628 | 0.536 | 0.594 |
| CNN, mean pool | 0.572 [0.550, 0.595] | +0.016 [-0.021, +0.054] | 0.444 | 0.567 | 0.604 | 0.569 | 0.598 |
| CNN, attention pool | 0.604 [0.580, 0.628] | +0.048 [+0.011, +0.087] | 0.496 | 0.561 | 0.639 | 0.595 | 0.608 |
| small ResNet, mean pool | 0.637 [0.613, 0.658] | +0.081 [+0.042, +0.122] | 0.588 | 0.605 | 0.622 | 0.611 | 0.654 |
| small ResNet, attention pool | 0.627 [0.601, 0.650] | +0.070 [+0.033, +0.111] | 0.590 | 0.554 | 0.636 | 0.585 | 0.590 |
| small ResNet (no max pool), mean pool | 0.607 [0.580, 0.635] | +0.051 [+0.012, +0.094] | 0.532 | 0.600 | 0.573 | 0.629 | 0.615 |
| small ResNet (no max pool), attention pool | 0.592 [0.567, 0.615] | +0.036 [-0.002, +0.074] | 0.591 | 0.572 | 0.615 | 0.544 | 0.594 |
| ResNet-18, mean pool | 0.602 [0.575, 0.630] | +0.046 [+0.006, +0.087] | 0.500 | 0.660 | 0.595 | 0.572 | 0.638 |
| ResNet-18, attention pool | 0.621 [0.594, 0.646] | +0.065 [+0.026, +0.103] | 0.594 | 0.624 | 0.592 | 0.604 | 0.583 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.448 | 0.444 | 0.481 | 0.590 |
| CNN, mean pool | 0.428 | 0.500 | 0.526 | 0.523 |
| CNN, attention pool | 0.464 | 0.555 | 0.581 | 0.528 |
| small ResNet, mean pool | 0.513 | 0.557 | 0.563 | 0.569 |
| small ResNet, attention pool | 0.593 | 0.586 | 0.553 | 0.539 |
| small ResNet (no max pool), mean pool | 0.447 | 0.588 | 0.553 | 0.593 |
| small ResNet (no max pool), attention pool | 0.454 | 0.563 | 0.556 | 0.514 |
| ResNet-18, mean pool | 0.467 | 0.555 | 0.575 | 0.549 |
| ResNet-18, attention pool | 0.554 | 0.543 | 0.561 | 0.549 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.573, 0.585, 0.607 | 0.588 ± 0.017 |
| CNN, attention pool | 0.592, 0.618, 0.559 | 0.590 ± 0.030 |
| small ResNet, mean pool | 0.608, 0.620, 0.604 | 0.611 ± 0.008 |
| small ResNet, attention pool | 0.611, 0.598, 0.633 | 0.614 ± 0.017 |
| small ResNet (no max pool), mean pool | 0.580, 0.612, 0.589 | 0.594 ± 0.017 |
| small ResNet (no max pool), attention pool | 0.579, 0.588, 0.564 | 0.577 ± 0.012 |
| ResNet-18, mean pool | 0.598, 0.619, 0.555 | 0.590 ± 0.033 |
| ResNet-18, attention pool | 0.628, 0.607, 0.596 | 0.610 ± 0.016 |

Median best epoch: CNN, mean pool 10, CNN, attention pool 7, small ResNet, mean pool 7, small ResNet, attention pool 6, small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 8, ResNet-18, mean pool 15, ResNet-18, attention pool 16.
