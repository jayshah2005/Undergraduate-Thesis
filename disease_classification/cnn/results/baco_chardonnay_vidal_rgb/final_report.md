# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | — | 0.585 | 0.529 | 0.550 | 0.544 | 0.553 |
| CNN, mean pool | 0.560 [0.531, 0.589] | +0.017 [-0.010, +0.043] | 0.562 | 0.644 | 0.540 | 0.550 | 0.537 |
| CNN, attention pool | 0.572 [0.544, 0.598] | +0.029 [+0.004, +0.054] | 0.606 | 0.627 | 0.565 | 0.526 | 0.559 |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | +0.052 [+0.022, +0.083] | 0.577 | 0.615 | 0.545 | 0.581 | 0.527 |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | +0.022 [-0.002, +0.046] | 0.568 | 0.607 | 0.544 | 0.584 | 0.533 |
| small ResNet (no max pool), mean pool | 0.557 [0.527, 0.589] | +0.015 [-0.010, +0.040] | 0.582 | 0.609 | 0.519 | 0.563 | 0.542 |
| small ResNet (no max pool), attention pool | 0.552 [0.521, 0.581] | +0.009 [-0.017, +0.034] | 0.577 | 0.613 | 0.517 | 0.552 | 0.529 |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | +0.044 [+0.007, +0.080] | 0.557 | 0.628 | 0.540 | 0.509 | 0.571 |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | +0.068 [+0.033, +0.105] | 0.595 | 0.633 | 0.561 | 0.493 | 0.637 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.466 |
| CNN, mean pool | 0.454 | 0.456 | 0.466 |
| CNN, attention pool | 0.522 | 0.419 | 0.497 |
| small ResNet, mean pool | 0.524 | 0.525 | 0.446 |
| small ResNet, attention pool | 0.477 | 0.440 | 0.488 |
| small ResNet (no max pool), mean pool | 0.477 | 0.431 | 0.501 |
| small ResNet (no max pool), attention pool | 0.483 | 0.412 | 0.500 |
| ResNet-18, mean pool | 0.523 | 0.509 | 0.459 |
| ResNet-18, attention pool | 0.549 | 0.471 | 0.520 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.580, 0.548, 0.538 | 0.555 ± 0.022 |
| CNN, attention pool | 0.562, 0.538, 0.574 | 0.558 ± 0.018 |
| small ResNet, mean pool | 0.594, 0.547, 0.567 | 0.569 ± 0.023 |
| small ResNet, attention pool | 0.572, 0.546, 0.567 | 0.561 ± 0.014 |
| small ResNet (no max pool), mean pool | 0.592, 0.520, 0.565 | 0.559 ± 0.036 |
| small ResNet (no max pool), attention pool | 0.567, 0.525, 0.583 | 0.558 ± 0.030 |
| ResNet-18, mean pool | 0.591, 0.599, 0.570 | 0.587 ± 0.015 |
| ResNet-18, attention pool | 0.597, 0.592, 0.583 | 0.591 ± 0.007 |

Median best epoch: CNN, mean pool 9, CNN, attention pool 2, small ResNet, mean pool 7, small ResNet, attention pool 4, small ResNet (no max pool), mean pool 9, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 15, ResNet-18, attention pool 10.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | — | 0.616 | 0.575 | 0.693 | 0.547 | 0.661 |
| CNN, mean pool | 0.611 [0.585, 0.637] | +0.002 [-0.019, +0.024] | 0.608 | 0.580 | 0.663 | 0.568 | 0.661 |
| CNN, attention pool | 0.609 [0.583, 0.635] | +0.000 [-0.019, +0.018] | 0.606 | 0.582 | 0.660 | 0.576 | 0.662 |
| small ResNet, mean pool | 0.604 [0.579, 0.630] | -0.005 [-0.026, +0.016] | 0.595 | 0.568 | 0.662 | 0.569 | 0.655 |
| small ResNet, attention pool | 0.596 [0.568, 0.622] | -0.014 [-0.034, +0.008] | 0.594 | 0.573 | 0.668 | 0.549 | 0.633 |
| small ResNet (no max pool), mean pool | 0.607 [0.581, 0.633] | -0.003 [-0.021, +0.016] | 0.586 | 0.575 | 0.671 | 0.592 | 0.653 |
| small ResNet (no max pool), attention pool | 0.615 [0.590, 0.639] | +0.005 [-0.011, +0.021] | 0.619 | 0.560 | 0.705 | 0.585 | 0.655 |
| ResNet-18, mean pool | 0.605 [0.578, 0.632] | -0.004 [-0.036, +0.028] | 0.551 | 0.626 | 0.693 | 0.545 | 0.618 |
| ResNet-18, attention pool | 0.581 [0.557, 0.605] | -0.029 [-0.057, -0.000] | 0.559 | 0.534 | 0.616 | 0.522 | 0.663 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.441 | 0.610 | 0.548 |
| CNN, mean pool | 0.493 | 0.580 | 0.543 |
| CNN, attention pool | 0.473 | 0.592 | 0.532 |
| small ResNet, mean pool | 0.477 | 0.568 | 0.539 |
| small ResNet, attention pool | 0.480 | 0.559 | 0.530 |
| small ResNet (no max pool), mean pool | 0.461 | 0.594 | 0.545 |
| small ResNet (no max pool), attention pool | 0.471 | 0.602 | 0.536 |
| ResNet-18, mean pool | 0.511 | 0.546 | 0.550 |
| ResNet-18, attention pool | 0.487 | 0.528 | 0.532 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.612, 0.560, 0.608 | 0.593 ± 0.029 |
| CNN, attention pool | 0.615, 0.609, 0.601 | 0.608 ± 0.007 |
| small ResNet, mean pool | 0.590, 0.599, 0.600 | 0.596 ± 0.005 |
| small ResNet, attention pool | 0.597, 0.589, 0.605 | 0.597 ± 0.008 |
| small ResNet (no max pool), mean pool | 0.593, 0.586, 0.592 | 0.590 ± 0.004 |
| small ResNet (no max pool), attention pool | 0.609, 0.609, 0.586 | 0.601 ± 0.013 |
| ResNet-18, mean pool | 0.565, 0.615, 0.566 | 0.582 ± 0.029 |
| ResNet-18, attention pool | 0.580, 0.553, 0.582 | 0.572 ± 0.016 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 5, small ResNet, mean pool 6, small ResNet, attention pool 6, small ResNet (no max pool), mean pool 4, small ResNet (no max pool), attention pool 6, ResNet-18, mean pool 18, ResNet-18, attention pool 11.
