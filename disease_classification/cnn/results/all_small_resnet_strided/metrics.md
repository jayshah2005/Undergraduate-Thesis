# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | 0.586 [0.564, 0.606] | 0.638 [0.602, 0.673] | 0.559 [0.530, 0.585] | 0.426 [0.407, 0.445] | 0.511 [0.488, 0.533] |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | 0.576 [0.557, 0.596] | 0.701 [0.665, 0.736] | 0.512 [0.490, 0.534] | 0.425 [0.408, 0.442] | 0.529 [0.507, 0.550] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| small ResNet (no max pool), mean pool | 162 | 92 | 218 | 276 |
| small ResNet (no max pool), attention pool | 178 | 76 | 241 | 253 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| small ResNet (no max pool), mean pool | Baco | 0.528 | 0.504 | 0.701 | 0.355 | 0.452 | 0.549 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.458 | 0.452 | 0.689 | 0.227 | 0.459 | 0.551 |
| small ResNet (no max pool), mean pool | Vidal | 0.498 | 0.800 | 0.120 | 0.876 | 0.097 | 0.107 |
| small ResNet (no max pool), attention pool | Baco | 0.538 | 0.512 | 0.729 | 0.348 | 0.459 | 0.563 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.467 | 0.460 | 0.762 | 0.172 | 0.467 | 0.579 |
| small ResNet (no max pool), attention pool | Vidal | 0.544 | 0.756 | 0.280 | 0.809 | 0.140 | 0.187 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | 0.634 [0.614, 0.655] | 0.626 [0.599, 0.652] | 0.649 [0.614, 0.682] | 0.756 [0.737, 0.775] | 0.685 [0.664, 0.704] |
| small ResNet (no max pool), mean pool | 0.673 [0.652, 0.692] | 0.649 [0.630, 0.667] | 0.586 [0.562, 0.608] | 0.759 [0.726, 0.792] | 0.808 [0.787, 0.830] | 0.679 [0.660, 0.698] |
| small ResNet (no max pool), attention pool | 0.681 [0.662, 0.700] | 0.667 [0.649, 0.684] | 0.629 [0.607, 0.651] | 0.734 [0.704, 0.764] | 0.804 [0.785, 0.823] | 0.706 [0.688, 0.722] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 237 | 128 | 237 |
| small ResNet (no max pool), mean pool | 371 | 262 | 88 | 277 |
| small ResNet (no max pool), attention pool | 398 | 235 | 97 | 268 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.437 | 0.419 | 0.237 | 0.637 | 0.438 | 0.308 |
| logistic regression | Cab Franc | 0.473 | 0.808 | 0.834 | 0.111 | 0.962 | 0.893 |
| logistic regression | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| logistic regression | Vidal | 0.515 | 0.644 | 0.882 | 0.148 | 0.683 | 0.770 |
| small ResNet (no max pool), mean pool | Baco | 0.485 | 0.448 | 0.059 | 0.912 | 0.444 | 0.105 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.518 | 0.896 | 0.925 | 0.111 | 0.965 | 0.945 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.533 | 0.640 | 0.170 | 0.895 | 0.469 | 0.250 |
| small ResNet (no max pool), mean pool | Vidal | 0.543 | 0.612 | 0.740 | 0.346 | 0.702 | 0.720 |
| small ResNet (no max pool), attention pool | Baco | 0.477 | 0.444 | 0.096 | 0.858 | 0.448 | 0.159 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.520 | 0.900 | 0.929 | 0.111 | 0.966 | 0.947 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.538 | 0.660 | 0.125 | 0.951 | 0.579 | 0.206 |
| small ResNet (no max pool), attention pool | Vidal | 0.543 | 0.664 | 0.888 | 0.198 | 0.698 | 0.781 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | 0.581 [0.555, 0.607] | 0.602 [0.573, 0.631] | 0.509 [0.451, 0.562] | 0.807 [0.788, 0.827] | 0.690 [0.667, 0.712] |
| small ResNet (no max pool), mean pool | 0.598 [0.574, 0.621] | 0.596 [0.574, 0.617] | 0.595 [0.569, 0.618] | 0.602 [0.562, 0.642] | 0.836 [0.820, 0.850] | 0.695 [0.674, 0.714] |
| small ResNet (no max pool), attention pool | 0.580 [0.553, 0.607] | 0.566 [0.542, 0.589] | 0.554 [0.527, 0.580] | 0.606 [0.562, 0.655] | 0.828 [0.809, 0.847] | 0.664 [0.641, 0.686] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 465 | 307 | 111 | 115 |
| small ResNet (no max pool), mean pool | 459 | 313 | 90 | 136 |
| small ResNet (no max pool), attention pool | 428 | 344 | 89 | 137 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.431 | 0.480 | 0.524 | 0.339 | 0.717 | 0.606 |
| logistic regression | Cab Franc | 0.435 | 0.736 | 0.759 | 0.111 | 0.958 | 0.847 |
| logistic regression | Chardonnay | 0.491 | 0.436 | 0.294 | 0.689 | 0.627 | 0.400 |
| logistic regression | Vidal | 0.609 | 0.672 | 0.747 | 0.471 | 0.791 | 0.768 |
| small ResNet (no max pool), mean pool | Baco | 0.553 | 0.540 | 0.529 | 0.576 | 0.800 | 0.637 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.639 | 0.820 | 0.834 | 0.444 | 0.976 | 0.899 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.523 | 0.436 | 0.212 | 0.833 | 0.694 | 0.325 |
| small ResNet (no max pool), mean pool | Vidal | 0.510 | 0.588 | 0.681 | 0.338 | 0.734 | 0.707 |
| small ResNet (no max pool), attention pool | Baco | 0.407 | 0.371 | 0.339 | 0.475 | 0.674 | 0.451 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.607 | 0.860 | 0.880 | 0.333 | 0.972 | 0.924 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.568 | 0.484 | 0.269 | 0.867 | 0.782 | 0.400 |
| small ResNet (no max pool), attention pool | Vidal | 0.505 | 0.548 | 0.599 | 0.412 | 0.732 | 0.659 |
