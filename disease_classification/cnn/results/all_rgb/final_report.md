# Disease: whole-leaf CNN on all varieties, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| CNN, mean pool | 0.560 [0.531, 0.589] | +0.017 [-0.010, +0.043] | 0.562 | 0.644 | 0.540 | 0.550 | 0.537 |
| CNN, attention pool | 0.572 [0.544, 0.598] | +0.029 [+0.004, +0.054] | 0.606 | 0.627 | 0.565 | 0.526 | 0.559 |
| CNN, mean ∥ max pool | 0.575 [0.547, 0.600] | +0.032 [-0.000, +0.062] | 0.583 | 0.646 | 0.546 | 0.541 | 0.524 |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | +0.052 [+0.022, +0.083] | 0.577 | 0.615 | 0.545 | 0.581 | 0.527 |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | +0.022 [-0.002, +0.046] | 0.568 | 0.607 | 0.544 | 0.584 | 0.533 |
| small ResNet, mean ∥ max pool | 0.577 [0.553, 0.601] | +0.035 [-0.000, +0.069] | 0.537 | 0.646 | 0.535 | 0.586 | 0.524 |
| small ResNet (no max pool), mean pool | 0.557 [0.527, 0.589] | +0.015 [-0.010, +0.040] | 0.582 | 0.609 | 0.519 | 0.563 | 0.542 |
| small ResNet (no max pool), attention pool | 0.552 [0.521, 0.581] | +0.009 [-0.017, +0.034] | 0.577 | 0.613 | 0.517 | 0.552 | 0.529 |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | +0.044 [+0.007, +0.080] | 0.557 | 0.628 | 0.540 | 0.509 | 0.571 |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | +0.068 [+0.033, +0.105] | 0.595 | 0.633 | 0.561 | 0.493 | 0.637 |
| CNN + ECA, mean pool | 0.566 [0.537, 0.594] | +0.024 [-0.001, +0.047] | 0.567 | 0.627 | 0.560 | 0.543 | 0.541 |
| CNN + ECA, attention pool | 0.587 [0.558, 0.614] | +0.045 [+0.018, +0.071] | 0.595 | 0.634 | 0.553 | 0.558 | 0.574 |
| CNN + CBAM, mean pool | 0.590 [0.561, 0.620] | +0.048 [+0.017, +0.080] | 0.608 | 0.636 | 0.597 | 0.571 | 0.529 |
| CNN + CBAM, attention pool | 0.602 [0.575, 0.630] | +0.059 [+0.032, +0.088] | 0.594 | 0.621 | 0.632 | 0.556 | 0.569 |
| small ResNet + ECA, mean pool | 0.551 [0.522, 0.578] | +0.008 [-0.019, +0.034] | 0.573 | 0.609 | 0.575 | 0.506 | 0.547 |
| small ResNet + ECA, attention pool | 0.612 [0.584, 0.637] | +0.069 [+0.040, +0.098] | 0.594 | 0.646 | 0.559 | 0.627 | 0.550 |
| small ResNet + CBAM, mean pool | 0.580 [0.550, 0.610] | +0.037 [+0.010, +0.066] | 0.609 | 0.615 | 0.594 | 0.544 | 0.554 |
| small ResNet + CBAM, attention pool | 0.579 [0.551, 0.606] | +0.037 [+0.007, +0.068] | 0.613 | 0.617 | 0.540 | 0.580 | 0.549 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| CNN, mean pool | 0.454 | 0.456 | 0.466 |
| CNN, attention pool | 0.522 | 0.419 | 0.497 |
| CNN, mean ∥ max pool | 0.503 | 0.449 | 0.481 |
| small ResNet, mean pool | 0.524 | 0.525 | 0.446 |
| small ResNet, attention pool | 0.477 | 0.440 | 0.488 |
| small ResNet, mean ∥ max pool | 0.509 | 0.486 | 0.486 |
| small ResNet (no max pool), mean pool | 0.477 | 0.431 | 0.501 |
| small ResNet (no max pool), attention pool | 0.483 | 0.412 | 0.500 |
| ResNet-18, mean pool | 0.523 | 0.509 | 0.459 |
| ResNet-18, attention pool | 0.549 | 0.471 | 0.520 |
| CNN + ECA, mean pool | 0.479 | 0.436 | 0.484 |
| CNN + ECA, attention pool | 0.512 | 0.452 | 0.480 |
| CNN + CBAM, mean pool | 0.534 | 0.446 | 0.494 |
| CNN + CBAM, attention pool | 0.505 | 0.504 | 0.502 |
| small ResNet + ECA, mean pool | 0.468 | 0.420 | 0.481 |
| small ResNet + ECA, attention pool | 0.538 | 0.499 | 0.492 |
| small ResNet + CBAM, mean pool | 0.498 | 0.439 | 0.508 |
| small ResNet + CBAM, attention pool | 0.503 | 0.462 | 0.491 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.580, 0.548, 0.538 | 0.555 ± 0.022 |
| CNN, attention pool | 0.562, 0.538, 0.574 | 0.558 ± 0.018 |
| CNN, mean ∥ max pool | 0.565, 0.567, 0.566 | 0.566 ± 0.001 |
| small ResNet, mean pool | 0.594, 0.547, 0.567 | 0.569 ± 0.023 |
| small ResNet, attention pool | 0.572, 0.546, 0.567 | 0.561 ± 0.014 |
| small ResNet, mean ∥ max pool | 0.536, 0.582, 0.551 | 0.556 ± 0.024 |
| small ResNet (no max pool), mean pool | 0.592, 0.520, 0.565 | 0.559 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.567, 0.525, 0.583 | 0.558 ± 0.030 |
| ResNet-18, mean pool | 0.591, 0.599, 0.570 | 0.587 ± 0.015 |
| ResNet-18, attention pool | 0.597, 0.592, 0.583 | 0.591 ± 0.007 |
| CNN + ECA, mean pool | 0.553, 0.570, 0.575 | 0.566 ± 0.011 |
| CNN + ECA, attention pool | 0.540, 0.583, 0.616 | 0.580 ± 0.038 |
| CNN + CBAM, mean pool | 0.569, 0.588, 0.609 | 0.588 ± 0.020 |
| CNN + CBAM, attention pool | 0.599, 0.603, 0.585 | 0.596 ± 0.010 |
| small ResNet + ECA, mean pool | 0.583, 0.536, 0.549 | 0.556 ± 0.024 |
| small ResNet + ECA, attention pool | 0.585, 0.562, 0.594 | 0.580 ± 0.016 |
| small ResNet + CBAM, mean pool | 0.591, 0.561, 0.569 | 0.574 ± 0.016 |
| small ResNet + CBAM, attention pool | 0.582, 0.579, 0.591 | 0.584 ± 0.006 |

Median best epoch: CNN, mean pool 9, CNN, attention pool 2, CNN, mean ∥ max pool 5, small ResNet, mean pool 7, small ResNet, attention pool 4, small ResNet, mean ∥ max pool 7, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 15, ResNet-18, attention pool 10, CNN + ECA, mean pool 3, CNN + ECA, attention pool 13, CNN + CBAM, mean pool 8, CNN + CBAM, attention pool 7, small ResNet + ECA, mean pool 7, small ResNet + ECA, attention pool 4, small ResNet + CBAM, mean pool 3, small ResNet + CBAM, attention pool 10.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.015 [-0.013, +0.040] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | -0.018 [-0.045, +0.008] |
| CNN + ECA, mean pool | CNN, mean pool | +0.006 [-0.010, +0.023] |
| CNN + ECA, attention pool | CNN, attention pool | +0.015 [-0.004, +0.033] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.031 [+0.002, +0.061] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.030 [+0.002, +0.059] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | -0.044 [-0.066, -0.022] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.047 [+0.026, +0.069] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | -0.015 [-0.040, +0.010] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.015 [-0.011, +0.042] |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | — | 0.621 | 0.638 | 0.709 | 0.648 | 0.642 |
| CNN, mean pool | 0.660 [0.640, 0.680] | +0.016 [-0.005, +0.036] | 0.656 | 0.649 | 0.638 | 0.690 | 0.683 |
| CNN, attention pool | 0.666 [0.645, 0.686] | +0.022 [-0.000, +0.044] | 0.663 | 0.642 | 0.695 | 0.675 | 0.650 |
| CNN, mean ∥ max pool | 0.668 [0.650, 0.687] | +0.024 [+0.003, +0.046] | 0.667 | 0.668 | 0.646 | 0.692 | 0.688 |
| small ResNet, mean pool | 0.659 [0.641, 0.678] | +0.015 [-0.005, +0.035] | 0.656 | 0.658 | 0.635 | 0.688 | 0.681 |
| small ResNet, attention pool | 0.668 [0.646, 0.688] | +0.024 [+0.003, +0.043] | 0.640 | 0.642 | 0.699 | 0.681 | 0.687 |
| small ResNet, mean ∥ max pool | 0.666 [0.647, 0.685] | +0.022 [+0.000, +0.044] | 0.639 | 0.675 | 0.677 | 0.652 | 0.703 |
| small ResNet (no max pool), mean pool | 0.653 [0.634, 0.672] | +0.009 [-0.010, +0.028] | 0.649 | 0.668 | 0.637 | 0.681 | 0.650 |
| small ResNet (no max pool), attention pool | 0.659 [0.641, 0.678] | +0.015 [-0.004, +0.033] | 0.645 | 0.666 | 0.670 | 0.692 | 0.650 |
| ResNet-18, mean pool | 0.665 [0.643, 0.685] | +0.020 [-0.004, +0.044] | 0.627 | 0.641 | 0.711 | 0.663 | 0.679 |
| ResNet-18, attention pool | 0.675 [0.655, 0.694] | +0.031 [+0.008, +0.054] | 0.650 | 0.677 | 0.671 | 0.679 | 0.698 |
| CNN + ECA, mean pool | 0.661 [0.641, 0.680] | +0.017 [-0.006, +0.037] | 0.658 | 0.664 | 0.642 | 0.671 | 0.679 |
| CNN + ECA, attention pool | 0.672 [0.651, 0.691] | +0.028 [+0.005, +0.050] | 0.668 | 0.664 | 0.694 | 0.679 | 0.650 |
| CNN + CBAM, mean pool | 0.678 [0.658, 0.698] | +0.034 [+0.012, +0.056] | 0.656 | 0.667 | 0.718 | 0.683 | 0.678 |
| CNN + CBAM, attention pool | 0.672 [0.651, 0.693] | +0.028 [+0.004, +0.051] | 0.657 | 0.676 | 0.677 | 0.677 | 0.681 |
| small ResNet + ECA, mean pool | 0.679 [0.658, 0.699] | +0.035 [+0.014, +0.056] | 0.663 | 0.670 | 0.695 | 0.690 | 0.689 |
| small ResNet + ECA, attention pool | 0.676 [0.655, 0.697] | +0.031 [+0.010, +0.052] | 0.658 | 0.633 | 0.718 | 0.692 | 0.695 |
| small ResNet + CBAM, mean pool | 0.676 [0.654, 0.696] | +0.031 [+0.008, +0.053] | 0.672 | 0.657 | 0.694 | 0.696 | 0.653 |
| small ResNet + CBAM, attention pool | 0.673 [0.653, 0.693] | +0.029 [+0.006, +0.052] | 0.641 | 0.662 | 0.696 | 0.677 | 0.699 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.461 | 0.499 | 0.573 | 0.540 |
| CNN, mean pool | 0.500 | 0.493 | 0.525 | 0.503 |
| CNN, attention pool | 0.492 | 0.495 | 0.539 | 0.512 |
| CNN, mean ∥ max pool | 0.516 | 0.502 | 0.525 | 0.510 |
| small ResNet, mean pool | 0.497 | 0.502 | 0.531 | 0.492 |
| small ResNet, attention pool | 0.508 | 0.485 | 0.542 | 0.521 |
| small ResNet, mean ∥ max pool | 0.503 | 0.497 | 0.517 | 0.521 |
| small ResNet (no max pool), mean pool | 0.494 | 0.487 | 0.539 | 0.487 |
| small ResNet (no max pool), attention pool | 0.509 | 0.479 | 0.536 | 0.512 |
| ResNet-18, mean pool | 0.480 | 0.516 | 0.537 | 0.525 |
| ResNet-18, attention pool | 0.520 | 0.527 | 0.564 | 0.487 |
| CNN + ECA, mean pool | 0.494 | 0.502 | 0.527 | 0.500 |
| CNN + ECA, attention pool | 0.481 | 0.508 | 0.542 | 0.526 |
| CNN + CBAM, mean pool | 0.501 | 0.529 | 0.537 | 0.522 |
| CNN + CBAM, attention pool | 0.477 | 0.524 | 0.533 | 0.534 |
| small ResNet + ECA, mean pool | 0.510 | 0.497 | 0.534 | 0.549 |
| small ResNet + ECA, attention pool | 0.507 | 0.483 | 0.534 | 0.559 |
| small ResNet + CBAM, mean pool | 0.495 | 0.508 | 0.524 | 0.552 |
| small ResNet + CBAM, attention pool | 0.486 | 0.504 | 0.531 | 0.532 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.666, 0.666, 0.653 | 0.662 ± 0.007 |
| CNN, attention pool | 0.663, 0.660, 0.668 | 0.664 ± 0.004 |
| CNN, mean ∥ max pool | 0.649, 0.678, 0.658 | 0.662 ± 0.015 |
| small ResNet, mean pool | 0.678, 0.665, 0.649 | 0.664 ± 0.015 |
| small ResNet, attention pool | 0.669, 0.653, 0.659 | 0.660 ± 0.008 |
| small ResNet, mean ∥ max pool | 0.659, 0.664, 0.652 | 0.658 ± 0.006 |
| small ResNet (no max pool), mean pool | 0.659, 0.660, 0.646 | 0.655 ± 0.008 |
| small ResNet (no max pool), attention pool | 0.652, 0.662, 0.649 | 0.654 ± 0.007 |
| ResNet-18, mean pool | 0.659, 0.659, 0.654 | 0.657 ± 0.003 |
| ResNet-18, attention pool | 0.649, 0.660, 0.654 | 0.654 ± 0.006 |
| CNN + ECA, mean pool | 0.654, 0.670, 0.673 | 0.666 ± 0.010 |
| CNN + ECA, attention pool | 0.659, 0.678, 0.678 | 0.672 ± 0.011 |
| CNN + CBAM, mean pool | 0.685, 0.665, 0.655 | 0.668 ± 0.015 |
| CNN + CBAM, attention pool | 0.681, 0.653, 0.662 | 0.665 ± 0.014 |
| small ResNet + ECA, mean pool | 0.667, 0.664, 0.665 | 0.665 ± 0.001 |
| small ResNet + ECA, attention pool | 0.676, 0.654, 0.670 | 0.666 ± 0.011 |
| small ResNet + CBAM, mean pool | 0.676, 0.674, 0.677 | 0.676 ± 0.001 |
| small ResNet + CBAM, attention pool | 0.675, 0.672, 0.670 | 0.672 ± 0.002 |

Median best epoch: CNN, mean pool 13, CNN, attention pool 11, CNN, mean ∥ max pool 15, small ResNet, mean pool 19, small ResNet, attention pool 17, small ResNet, mean ∥ max pool 15, small ResNet (no max pool), mean pool 14, small ResNet (no max pool), attention pool 15, ResNet-18, mean pool 15, ResNet-18, attention pool 20, CNN + ECA, mean pool 17, CNN + ECA, attention pool 17, CNN + CBAM, mean pool 15, CNN + CBAM, attention pool 17, small ResNet + ECA, mean pool 17, small ResNet + ECA, attention pool 12, small ResNet + CBAM, mean pool 16, small ResNet + CBAM, attention pool 14.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | +0.008 [-0.006, +0.023] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.007 [-0.008, +0.022] |
| CNN + ECA, mean pool | CNN, mean pool | +0.001 [-0.009, +0.010] |
| CNN + ECA, attention pool | CNN, attention pool | +0.006 [-0.006, +0.017] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.018 [-0.001, +0.037] |
| CNN + CBAM, attention pool | CNN, attention pool | +0.006 [-0.011, +0.023] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.020 [+0.004, +0.035] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.008 [-0.006, +0.022] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.016 [-0.003, +0.035] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.005 [-0.011, +0.021] |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | — | 0.576 | 0.542 | 0.655 | 0.591 | 0.577 |
| CNN, mean pool | 0.591 [0.569, 0.614] | +0.009 [-0.022, +0.036] | 0.477 | 0.560 | 0.577 | 0.645 | 0.625 |
| CNN, attention pool | 0.594 [0.573, 0.616] | +0.011 [-0.017, +0.038] | 0.449 | 0.561 | 0.607 | 0.623 | 0.625 |
| small ResNet, mean pool | 0.597 [0.574, 0.620] | +0.014 [-0.014, +0.041] | 0.484 | 0.567 | 0.590 | 0.623 | 0.618 |
| small ResNet, attention pool | 0.598 [0.577, 0.619] | +0.015 [-0.012, +0.041] | 0.530 | 0.561 | 0.591 | 0.630 | 0.633 |
| small ResNet (no max pool), mean pool | 0.587 [0.564, 0.611] | +0.005 [-0.019, +0.029] | 0.563 | 0.559 | 0.577 | 0.615 | 0.591 |
| small ResNet (no max pool), attention pool | 0.591 [0.572, 0.613] | +0.009 [-0.015, +0.032] | 0.531 | 0.543 | 0.595 | 0.619 | 0.619 |
| ResNet-18, mean pool | 0.640 [0.617, 0.662] | +0.058 [+0.029, +0.087] | 0.652 | 0.566 | 0.614 | 0.615 | 0.630 |
| ResNet-18, attention pool | 0.620 [0.595, 0.646] | +0.037 [+0.008, +0.066] | 0.664 | 0.558 | 0.611 | 0.555 | 0.639 |

Fold k tests on row Rk of every variety (4 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| logistic regression | 0.440 | 0.497 | 0.542 | 0.532 |
| CNN, mean pool | 0.482 | 0.536 | 0.486 | 0.509 |
| CNN, attention pool | 0.497 | 0.485 | 0.499 | 0.504 |
| small ResNet, mean pool | 0.508 | 0.491 | 0.497 | 0.507 |
| small ResNet, attention pool | 0.492 | 0.470 | 0.505 | 0.521 |
| small ResNet (no max pool), mean pool | 0.482 | 0.409 | 0.518 | 0.497 |
| small ResNet (no max pool), attention pool | 0.476 | 0.464 | 0.518 | 0.511 |
| ResNet-18, mean pool | 0.600 | 0.677 | 0.556 | 0.491 |
| ResNet-18, attention pool | 0.564 | 0.592 | 0.565 | 0.508 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.621, 0.595, 0.591 | 0.602 ± 0.016 |
| CNN, attention pool | 0.596, 0.600, 0.590 | 0.595 ± 0.005 |
| small ResNet, mean pool | 0.604, 0.597, 0.598 | 0.600 ± 0.003 |
| small ResNet, attention pool | 0.589, 0.595, 0.611 | 0.598 ± 0.011 |
| small ResNet (no max pool), mean pool | 0.602, 0.592, 0.601 | 0.598 ± 0.006 |
| small ResNet (no max pool), attention pool | 0.585, 0.576, 0.601 | 0.588 ± 0.013 |
| ResNet-18, mean pool | 0.599, 0.629, 0.632 | 0.620 ± 0.018 |
| ResNet-18, attention pool | 0.636, 0.608, 0.570 | 0.605 ± 0.033 |

Median best epoch: CNN, mean pool 16, CNN, attention pool 17, small ResNet, mean pool 11, small ResNet, attention pool 15, small ResNet (no max pool), mean pool 17, small ResNet (no max pool), attention pool 19, ResNet-18, mean pool 22, ResNet-18, attention pool 11.
