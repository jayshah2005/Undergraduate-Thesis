# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| CNN, mean pool | 0.619 [0.599, 0.639] | +0.019 [-0.009, +0.046] | 0.554 | 0.596 | 0.699 | 0.594 | 0.626 |
| CNN, attention pool | 0.606 [0.584, 0.629] | +0.006 [-0.024, +0.035] | 0.542 | 0.589 | 0.694 | 0.569 | 0.636 |
| CNN, mean ∥ max pool | 0.611 [0.586, 0.634] | +0.011 [-0.021, +0.042] | 0.591 | 0.592 | 0.670 | 0.557 | 0.642 |
| small ResNet, mean pool | 0.595 [0.571, 0.619] | -0.004 [-0.038, +0.028] | 0.558 | 0.586 | 0.623 | 0.571 | 0.630 |
| small ResNet, attention pool | 0.594 [0.571, 0.616] | -0.006 [-0.038, +0.025] | 0.522 | 0.626 | 0.599 | 0.578 | 0.650 |
| small ResNet, mean ∥ max pool | 0.619 [0.598, 0.640] | +0.020 [-0.012, +0.051] | 0.602 | 0.599 | 0.664 | 0.543 | 0.653 |
| small ResNet (no max pool), mean pool | 0.624 [0.599, 0.646] | +0.024 [-0.008, +0.054] | 0.551 | 0.637 | 0.651 | 0.608 | 0.641 |
| small ResNet (no max pool), attention pool | 0.590 [0.566, 0.614] | -0.009 [-0.042, +0.021] | 0.544 | 0.607 | 0.662 | 0.544 | 0.632 |
| ResNet-18, mean pool | 0.602 [0.580, 0.623] | +0.002 [-0.031, +0.032] | 0.570 | 0.589 | 0.621 | 0.572 | 0.641 |
| ResNet-18, attention pool | 0.591 [0.567, 0.615] | -0.009 [-0.043, +0.025] | 0.546 | 0.603 | 0.609 | 0.579 | 0.633 |
| CNN + ECA, mean pool | 0.580 [0.556, 0.603] | -0.020 [-0.052, +0.012] | 0.530 | 0.574 | 0.647 | 0.529 | 0.638 |
| CNN + ECA, attention pool | 0.609 [0.589, 0.627] | +0.009 [-0.020, +0.037] | 0.597 | 0.574 | 0.613 | 0.577 | 0.635 |
| CNN + CBAM, mean pool | 0.623 [0.602, 0.644] | +0.023 [-0.007, +0.053] | 0.590 | 0.633 | 0.643 | 0.571 | 0.632 |
| CNN + CBAM, attention pool | 0.589 [0.565, 0.612] | -0.011 [-0.043, +0.020] | 0.538 | 0.580 | 0.626 | 0.597 | 0.630 |
| small ResNet + ECA, mean pool | 0.609 [0.586, 0.632] | +0.009 [-0.021, +0.040] | 0.576 | 0.593 | 0.654 | 0.568 | 0.633 |
| small ResNet + ECA, attention pool | 0.626 [0.605, 0.646] | +0.027 [-0.003, +0.054] | 0.590 | 0.603 | 0.670 | 0.598 | 0.626 |
| small ResNet + CBAM, mean pool | 0.607 [0.585, 0.631] | +0.008 [-0.022, +0.038] | 0.557 | 0.592 | 0.651 | 0.584 | 0.627 |
| small ResNet + CBAM, attention pool | 0.623 [0.597, 0.648] | +0.023 [-0.010, +0.057] | 0.576 | 0.635 | 0.679 | 0.584 | 0.645 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| CNN, mean pool | 0.497 | 0.557 | 0.542 |
| CNN, attention pool | 0.475 | 0.558 | 0.547 |
| CNN, mean ∥ max pool | 0.491 | 0.548 | 0.540 |
| small ResNet, mean pool | 0.481 | 0.545 | 0.553 |
| small ResNet, attention pool | 0.487 | 0.541 | 0.542 |
| small ResNet, mean ∥ max pool | 0.514 | 0.557 | 0.564 |
| small ResNet (no max pool), mean pool | 0.530 | 0.565 | 0.530 |
| small ResNet (no max pool), attention pool | 0.451 | 0.527 | 0.537 |
| ResNet-18, mean pool | 0.501 | 0.535 | 0.522 |
| ResNet-18, attention pool | 0.489 | 0.542 | 0.531 |
| CNN + ECA, mean pool | 0.441 | 0.548 | 0.530 |
| CNN + ECA, attention pool | 0.503 | 0.553 | 0.544 |
| CNN + CBAM, mean pool | 0.527 | 0.560 | 0.548 |
| CNN + CBAM, attention pool | 0.463 | 0.548 | 0.527 |
| small ResNet + ECA, mean pool | 0.492 | 0.563 | 0.543 |
| small ResNet + ECA, attention pool | 0.521 | 0.543 | 0.576 |
| small ResNet + CBAM, mean pool | 0.481 | 0.564 | 0.562 |
| small ResNet + CBAM, attention pool | 0.529 | 0.579 | 0.554 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.609, 0.595, 0.609 | 0.604 ± 0.008 |
| CNN, attention pool | 0.605, 0.583, 0.600 | 0.596 ± 0.012 |
| CNN, mean ∥ max pool | 0.588, 0.623, 0.616 | 0.609 ± 0.018 |
| small ResNet, mean pool | 0.578, 0.623, 0.572 | 0.591 ± 0.028 |
| small ResNet, attention pool | 0.591, 0.592, 0.597 | 0.593 ± 0.004 |
| small ResNet, mean ∥ max pool | 0.576, 0.612, 0.616 | 0.601 ± 0.022 |
| small ResNet (no max pool), mean pool | 0.618, 0.604, 0.598 | 0.606 ± 0.010 |
| small ResNet (no max pool), attention pool | 0.618, 0.587, 0.593 | 0.600 ± 0.016 |
| ResNet-18, mean pool | 0.569, 0.616, 0.574 | 0.586 ± 0.026 |
| ResNet-18, attention pool | 0.583, 0.547, 0.577 | 0.569 ± 0.019 |
| CNN + ECA, mean pool | 0.580, 0.596, 0.598 | 0.591 ± 0.010 |
| CNN + ECA, attention pool | 0.607, 0.596, 0.613 | 0.605 ± 0.008 |
| CNN + CBAM, mean pool | 0.588, 0.611, 0.614 | 0.604 ± 0.014 |
| CNN + CBAM, attention pool | 0.580, 0.559, 0.622 | 0.587 ± 0.032 |
| small ResNet + ECA, mean pool | 0.604, 0.609, 0.594 | 0.602 ± 0.008 |
| small ResNet + ECA, attention pool | 0.632, 0.613, 0.619 | 0.622 ± 0.010 |
| small ResNet + CBAM, mean pool | 0.595, 0.606, 0.631 | 0.611 ± 0.018 |
| small ResNet + CBAM, attention pool | 0.592, 0.614, 0.593 | 0.600 ± 0.012 |

Median best epoch: CNN, mean pool 13, CNN, attention pool 7, CNN, mean ∥ max pool 16, small ResNet, mean pool 10, small ResNet, attention pool 13, small ResNet, mean ∥ max pool 11, small ResNet (no max pool), mean pool 10, small ResNet (no max pool), attention pool 4, ResNet-18, mean pool 12, ResNet-18, attention pool 18, CNN + ECA, mean pool 9, CNN + ECA, attention pool 12, CNN + CBAM, mean pool 6, CNN + CBAM, attention pool 12, small ResNet + ECA, mean pool 13, small ResNet + ECA, attention pool 9, small ResNet + CBAM, mean pool 15, small ResNet + CBAM, attention pool 9.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| CNN, mean ∥ max pool | CNN, mean pool | -0.008 [-0.029, +0.013] |
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.024 [+0.003, +0.047] |
| CNN + ECA, mean pool | CNN, mean pool | -0.039 [-0.061, -0.017] |
| CNN + ECA, attention pool | CNN, attention pool | +0.003 [-0.018, +0.024] |
| CNN + CBAM, mean pool | CNN, mean pool | +0.004 [-0.017, +0.024] |
| CNN + CBAM, attention pool | CNN, attention pool | -0.018 [-0.038, +0.003] |
| small ResNet + ECA, mean pool | small ResNet, mean pool | +0.013 [-0.007, +0.035] |
| small ResNet + ECA, attention pool | small ResNet, attention pool | +0.032 [+0.011, +0.054] |
| small ResNet + CBAM, mean pool | small ResNet, mean pool | +0.012 [-0.004, +0.027] |
| small ResNet + CBAM, attention pool | small ResNet, attention pool | +0.029 [+0.007, +0.053] |
