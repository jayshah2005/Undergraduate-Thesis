# Disease: whole-leaf CNN on all varieties, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| CNN, mean pool | 0.586 [0.566, 0.606] | +0.014 [-0.021, +0.048] | 0.593 | 0.662 | 0.489 | 0.536 | 0.634 |
| CNN, attention pool | 0.597 [0.578, 0.618] | +0.026 [-0.010, +0.061] | 0.570 | 0.697 | 0.602 | 0.532 | 0.589 |
| CNN, mean ∥ max pool | 0.599 [0.579, 0.618] | +0.027 [-0.009, +0.061] | 0.647 | 0.643 | 0.535 | 0.551 | 0.633 |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | +0.014 [-0.022, +0.049] | 0.572 | 0.643 | 0.548 | 0.540 | 0.634 |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | +0.040 [+0.002, +0.075] | 0.489 | 0.668 | 0.611 | 0.522 | 0.645 |
| small ResNet, mean ∥ max pool | 0.595 [0.570, 0.620] | +0.023 [-0.017, +0.060] | 0.563 | 0.656 | 0.579 | 0.492 | 0.599 |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | +0.024 [-0.013, +0.058] | 0.624 | 0.676 | 0.494 | 0.566 | 0.599 |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | +0.030 [-0.004, +0.065] | 0.505 | 0.693 | 0.569 | 0.528 | 0.669 |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | +0.035 [-0.008, +0.075] | 0.541 | 0.652 | 0.623 | 0.561 | 0.598 |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | +0.002 [-0.042, +0.043] | 0.567 | 0.629 | 0.571 | 0.515 | 0.574 |
| CNN + ECA, mean pool | 0.603 [0.579, 0.625] | +0.031 [-0.005, +0.066] | 0.637 | 0.672 | 0.540 | 0.503 | 0.628 |
| CNN + ECA, attention pool | 0.590 [0.568, 0.611] | +0.018 [-0.018, +0.054] | 0.581 | 0.687 | 0.500 | 0.494 | 0.628 |
| CNN + CBAM, mean pool | 0.610 [0.591, 0.628] | +0.038 [+0.000, +0.073] | 0.625 | 0.691 | 0.551 | 0.533 | 0.683 |
| CNN + CBAM, attention pool | 0.624 [0.604, 0.643] | +0.052 [+0.016, +0.088] | 0.583 | 0.656 | 0.663 | 0.455 | 0.665 |
| small ResNet + ECA, mean pool | 0.607 [0.589, 0.625] | +0.035 [+0.001, +0.068] | 0.574 | 0.699 | 0.491 | 0.523 | 0.657 |
| small ResNet + ECA, attention pool | 0.580 [0.560, 0.599] | +0.008 [-0.028, +0.042] | 0.586 | 0.691 | 0.518 | 0.503 | 0.603 |
| small ResNet + CBAM, mean pool | 0.610 [0.591, 0.629] | +0.039 [+0.004, +0.072] | 0.585 | 0.689 | 0.516 | 0.555 | 0.672 |
| small ResNet + CBAM, attention pool | 0.583 [0.565, 0.601] | +0.012 [-0.023, +0.046] | 0.533 | 0.680 | 0.509 | 0.453 | 0.706 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| CNN, mean pool | 0.494 | 0.457 | 0.518 |
| CNN, attention pool | 0.516 | 0.454 | 0.468 |
| CNN, mean ∥ max pool | 0.510 | 0.450 | 0.495 |
| small ResNet, mean pool | 0.503 | 0.442 | 0.505 |
| small ResNet, attention pool | 0.518 | 0.520 | 0.502 |
| small ResNet, mean ∥ max pool | 0.462 | 0.502 | 0.479 |
| small ResNet (no max pool), mean pool | 0.480 | 0.497 | 0.463 |
| small ResNet (no max pool), attention pool | 0.524 | 0.475 | 0.496 |
| ResNet-18, mean pool | 0.548 | 0.502 | 0.493 |
| ResNet-18, attention pool | 0.507 | 0.419 | 0.514 |
| CNN + ECA, mean pool | 0.480 | 0.477 | 0.492 |
| CNN + ECA, attention pool | 0.478 | 0.503 | 0.488 |
| CNN + CBAM, mean pool | 0.497 | 0.456 | 0.496 |
| CNN + CBAM, attention pool | 0.497 | 0.508 | 0.525 |
| small ResNet + ECA, mean pool | 0.527 | 0.485 | 0.483 |
| small ResNet + ECA, attention pool | 0.489 | 0.450 | 0.526 |
| small ResNet + CBAM, mean pool | 0.527 | 0.488 | 0.500 |
| small ResNet + CBAM, attention pool | 0.501 | 0.431 | 0.510 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.572, 0.628, 0.551 | 0.584 ± 0.040 |
| CNN, attention pool | 0.591, 0.537, 0.578 | 0.569 ± 0.029 |
| CNN, mean ∥ max pool | 0.593, 0.587, 0.566 | 0.582 ± 0.014 |
| small ResNet, mean pool | 0.584, 0.575, 0.559 | 0.573 ± 0.012 |
| small ResNet, attention pool | 0.617, 0.609, 0.601 | 0.609 ± 0.008 |
| small ResNet, mean ∥ max pool | 0.567, 0.604, 0.575 | 0.582 ± 0.020 |
| small ResNet (no max pool), mean pool | 0.561, 0.566, 0.594 | 0.574 ± 0.018 |
| small ResNet (no max pool), attention pool | 0.534, 0.557, 0.595 | 0.562 ± 0.031 |
| ResNet-18, mean pool | 0.594, 0.609, 0.539 | 0.581 ± 0.037 |
| ResNet-18, attention pool | 0.540, 0.589, 0.570 | 0.566 ± 0.025 |
| CNN + ECA, mean pool | 0.592, 0.553, 0.585 | 0.577 ± 0.021 |
| CNN + ECA, attention pool | 0.573, 0.582, 0.556 | 0.570 ± 0.013 |
| CNN + CBAM, mean pool | 0.594, 0.567, 0.606 | 0.589 ± 0.020 |
| CNN + CBAM, attention pool | 0.609, 0.570, 0.608 | 0.596 ± 0.022 |
| small ResNet + ECA, mean pool | 0.626, 0.638, 0.555 | 0.607 ± 0.045 |
| small ResNet + ECA, attention pool | 0.562, 0.585, 0.566 | 0.571 ± 0.012 |
| small ResNet + CBAM, mean pool | 0.595, 0.597, 0.554 | 0.582 ± 0.024 |
| small ResNet + CBAM, attention pool | 0.565, 0.577, 0.561 | 0.567 ± 0.008 |

Median best epoch: CNN, mean pool 18, CNN, attention pool 14, CNN, mean ∥ max pool 16, small ResNet, mean pool 17, small ResNet, attention pool 18, small ResNet, mean ∥ max pool 17, small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 11, ResNet-18, mean pool 7, ResNet-18, attention pool 11, CNN + ECA, mean pool 17, CNN + ECA, attention pool 23, CNN + CBAM, mean pool 14, CNN + CBAM, attention pool 14, small ResNet + ECA, mean pool 22, small ResNet + ECA, attention pool 10, small ResNet + CBAM, mean pool 14, small ResNet + CBAM, attention pool 15.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.013 [-0.008, +0.034] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.010 [-0.016, +0.038] |
| CNN + ECA, mean pool | CNN, mean pool | +0.017 [-0.008, +0.043] |
| CNN + ECA, attention pool | CNN, attention pool | -0.008 [-0.033, +0.017] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.024 [+0.001, +0.046] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.026 [+0.003, +0.051] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.021 [+0.000, +0.041] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.032 [-0.058, -0.008] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.025 [+0.005, +0.044] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.029 [-0.056, -0.003] |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | — | 0.539 | 0.698 | 0.713 | 0.610 | 0.689 |
| CNN, mean pool | 0.691 [0.672, 0.709] | +0.044 [+0.019, +0.070] | 0.689 | 0.690 | 0.709 | 0.673 | 0.704 |
| CNN, attention pool | 0.666 [0.646, 0.685] | +0.020 [-0.007, +0.047] | 0.605 | 0.655 | 0.680 | 0.673 | 0.692 |
| CNN, mean ∥ max pool | 0.689 [0.671, 0.707] | +0.042 [+0.017, +0.069] | 0.688 | 0.667 | 0.755 | 0.646 | 0.690 |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | +0.036 [+0.008, +0.063] | 0.620 | 0.655 | 0.731 | 0.690 | 0.691 |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | +0.040 [+0.014, +0.065] | 0.652 | 0.664 | 0.748 | 0.669 | 0.695 |
| small ResNet, mean ∥ max pool | 0.675 [0.659, 0.691] | +0.028 [+0.002, +0.054] | 0.670 | 0.672 | 0.706 | 0.658 | 0.682 |
| small ResNet (no max pool), mean pool | 0.683 [0.665, 0.700] | +0.036 [+0.012, +0.061] | 0.651 | 0.683 | 0.741 | 0.662 | 0.680 |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | +0.027 [-0.000, +0.053] | 0.677 | 0.657 | 0.689 | 0.675 | 0.686 |
| ResNet-18, mean pool | 0.672 [0.647, 0.693] | +0.025 [-0.004, +0.053] | 0.640 | 0.684 | 0.644 | 0.681 | 0.705 |
| ResNet-18, attention pool | 0.667 [0.643, 0.689] | +0.020 [-0.010, +0.049] | 0.627 | 0.671 | 0.658 | 0.646 | 0.726 |
| CNN + ECA, mean pool | 0.686 [0.669, 0.702] | +0.039 [+0.014, +0.064] | 0.629 | 0.675 | 0.768 | 0.667 | 0.690 |
| CNN + ECA, attention pool | 0.681 [0.663, 0.698] | +0.034 [+0.009, +0.058] | 0.665 | 0.693 | 0.719 | 0.667 | 0.692 |
| CNN + CBAM, mean pool | 0.702 [0.683, 0.720] | +0.055 [+0.028, +0.081] | 0.668 | 0.687 | 0.740 | 0.679 | 0.736 |
| CNN + CBAM, attention pool | 0.701 [0.680, 0.721] | +0.054 [+0.028, +0.082] | 0.672 | 0.729 | 0.715 | 0.683 | 0.680 |
| small ResNet + ECA, mean pool | 0.683 [0.666, 0.701] | +0.037 [+0.011, +0.062] | 0.648 | 0.673 | 0.730 | 0.683 | 0.685 |
| small ResNet + ECA, attention pool | 0.680 [0.662, 0.699] | +0.033 [+0.007, +0.060] | 0.587 | 0.701 | 0.725 | 0.681 | 0.682 |
| small ResNet + CBAM, mean pool | 0.686 [0.666, 0.704] | +0.039 [+0.013, +0.065] | 0.678 | 0.691 | 0.705 | 0.679 | 0.691 |
| small ResNet + CBAM, attention pool | 0.678 [0.660, 0.695] | +0.031 [+0.006, +0.057] | 0.613 | 0.674 | 0.737 | 0.673 | 0.672 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.456 | 0.475 | 0.540 | 0.522 |
| CNN, mean pool | 0.499 | 0.522 | 0.555 | 0.539 |
| CNN, attention pool | 0.500 | 0.510 | 0.539 | 0.505 |
| CNN, mean ∥ max pool | 0.496 | 0.576 | 0.519 | 0.544 |
| small ResNet, mean pool | 0.504 | 0.512 | 0.555 | 0.537 |
| small ResNet, attention pool | 0.474 | 0.524 | 0.538 | 0.541 |
| small ResNet, mean ∥ max pool | 0.463 | 0.529 | 0.548 | 0.493 |
| small ResNet (no max pool), mean pool | 0.494 | 0.531 | 0.554 | 0.501 |
| small ResNet (no max pool), attention pool | 0.466 | 0.563 | 0.527 | 0.530 |
| ResNet-18, mean pool | 0.460 | 0.701 | 0.545 | 0.529 |
| ResNet-18, attention pool | 0.472 | 0.537 | 0.541 | 0.518 |
| CNN + ECA, mean pool | 0.489 | 0.545 | 0.569 | 0.519 |
| CNN + ECA, attention pool | 0.483 | 0.586 | 0.539 | 0.498 |
| CNN + CBAM, mean pool | 0.489 | 0.522 | 0.558 | 0.581 |
| CNN + CBAM, attention pool | 0.503 | 0.537 | 0.584 | 0.542 |
| small ResNet + ECA, mean pool | 0.463 | 0.533 | 0.547 | 0.529 |
| small ResNet + ECA, attention pool | 0.482 | 0.545 | 0.565 | 0.512 |
| small ResNet + CBAM, mean pool | 0.482 | 0.572 | 0.531 | 0.539 |
| small ResNet + CBAM, attention pool | 0.514 | 0.533 | 0.546 | 0.505 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.674, 0.664, 0.678 | 0.672 ± 0.007 |
| CNN, attention pool | 0.664, 0.658, 0.670 | 0.664 ± 0.006 |
| CNN, mean ∥ max pool | 0.665, 0.667, 0.648 | 0.660 ± 0.011 |
| small ResNet, mean pool | 0.684, 0.671, 0.675 | 0.677 ± 0.007 |
| small ResNet, attention pool | 0.689, 0.673, 0.656 | 0.672 ± 0.016 |
| small ResNet, mean ∥ max pool | 0.672, 0.668, 0.668 | 0.669 ± 0.002 |
| small ResNet (no max pool), mean pool | 0.669, 0.654, 0.684 | 0.669 ± 0.015 |
| small ResNet (no max pool), attention pool | 0.651, 0.645, 0.658 | 0.651 ± 0.007 |
| ResNet-18, mean pool | 0.654, 0.635, 0.672 | 0.654 ± 0.018 |
| ResNet-18, attention pool | 0.667, 0.679, 0.643 | 0.663 ± 0.018 |
| CNN + ECA, mean pool | 0.685, 0.643, 0.659 | 0.662 ± 0.021 |
| CNN + ECA, attention pool | 0.658, 0.664, 0.650 | 0.657 ± 0.007 |
| CNN + CBAM, mean pool | 0.670, 0.682, 0.687 | 0.680 ± 0.009 |
| CNN + CBAM, attention pool | 0.669, 0.687, 0.678 | 0.678 ± 0.009 |
| small ResNet + ECA, mean pool | 0.698, 0.678, 0.680 | 0.685 ± 0.011 |
| small ResNet + ECA, attention pool | 0.681, 0.668, 0.670 | 0.673 ± 0.007 |
| small ResNet + CBAM, mean pool | 0.674, 0.670, 0.666 | 0.670 ± 0.004 |
| small ResNet + CBAM, attention pool | 0.679, 0.682, 0.665 | 0.675 ± 0.009 |

Median best epoch: CNN, mean pool 10, CNN, attention pool 9, CNN, mean ∥ max pool 9, small ResNet, mean pool 11, small ResNet, attention pool 11, small ResNet, mean ∥ max pool 13, small ResNet (no max pool), mean pool 8, small ResNet (no max pool), attention pool 9, ResNet-18, mean pool 20, ResNet-18, attention pool 18, CNN + ECA, mean pool 10, CNN + ECA, attention pool 10, CNN + CBAM, mean pool 19, CNN + CBAM, attention pool 15, small ResNet + ECA, mean pool 17, small ResNet + ECA, attention pool 14, small ResNet + CBAM, mean pool 13, small ResNet + CBAM, attention pool 8.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.002 [-0.018, +0.015] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.007 [-0.026, +0.012] |
| CNN + ECA, mean pool | CNN, mean pool | -0.005 [-0.023, +0.013] |
| CNN + ECA, attention pool | CNN, attention pool | +0.014 [-0.006, +0.035] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.011 [-0.009, +0.030] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.035 [+0.012, +0.058] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.001 [-0.016, +0.019] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | -0.006 [-0.025, +0.013] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.003 [-0.016, +0.022] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | -0.008 [-0.026, +0.008] |

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
