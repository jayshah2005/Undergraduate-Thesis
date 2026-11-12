# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| CNN, mean pool | 0.604 [0.582, 0.626] | 0.594 [0.572, 0.615] | 0.638 [0.602, 0.673] | 0.571 [0.543, 0.599] | 0.433 [0.413, 0.454] | 0.516 [0.491, 0.541] |
| CNN, attention pool | 0.620 [0.598, 0.642] | 0.603 [0.583, 0.623] | 0.673 [0.634, 0.709] | 0.567 [0.545, 0.591] | 0.444 [0.425, 0.463] | 0.535 [0.512, 0.559] |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | 0.567 [0.547, 0.588] | 0.665 [0.630, 0.701] | 0.516 [0.492, 0.540] | 0.414 [0.397, 0.433] | 0.511 [0.487, 0.535] |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | 0.584 [0.561, 0.606] | 0.740 [0.701, 0.780] | 0.504 [0.478, 0.532] | 0.434 [0.415, 0.453] | 0.547 [0.524, 0.571] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| CNN, mean pool | 162 | 92 | 212 | 282 |
| CNN, attention pool | 171 | 83 | 214 | 280 |
| small ResNet, mean pool | 169 | 85 | 239 | 255 |
| small ResNet, attention pool | 188 | 66 | 245 | 249 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| CNN, mean pool | Baco | 0.519 | 0.488 | 0.748 | 0.291 | 0.444 | 0.557 |
| CNN, mean pool | Chardonnay | 0.476 | 0.472 | 0.639 | 0.312 | 0.470 | 0.542 |
| CNN, mean pool | Vidal | 0.527 | 0.820 | 0.160 | 0.893 | 0.143 | 0.151 |
| CNN, attention pool | Baco | 0.525 | 0.496 | 0.738 | 0.312 | 0.449 | 0.558 |
| CNN, attention pool | Chardonnay | 0.521 | 0.516 | 0.713 | 0.328 | 0.503 | 0.590 |
| CNN, attention pool | Vidal | 0.531 | 0.796 | 0.200 | 0.862 | 0.139 | 0.164 |
| small ResNet, mean pool | Baco | 0.510 | 0.472 | 0.785 | 0.234 | 0.438 | 0.562 |
| small ResNet, mean pool | Chardonnay | 0.445 | 0.440 | 0.664 | 0.227 | 0.450 | 0.536 |
| small ResNet, mean pool | Vidal | 0.509 | 0.788 | 0.160 | 0.858 | 0.111 | 0.131 |
| small ResNet, attention pool | Baco | 0.507 | 0.476 | 0.738 | 0.277 | 0.436 | 0.549 |
| small ResNet, attention pool | Chardonnay | 0.489 | 0.480 | 0.861 | 0.117 | 0.482 | 0.618 |
| small ResNet, attention pool | Vidal | 0.513 | 0.796 | 0.160 | 0.867 | 0.118 | 0.136 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | 0.634 [0.614, 0.655] | 0.626 [0.599, 0.652] | 0.649 [0.614, 0.682] | 0.756 [0.737, 0.775] | 0.685 [0.664, 0.704] |
| CNN, mean pool | 0.682 [0.665, 0.699] | 0.656 [0.639, 0.673] | 0.586 [0.564, 0.608] | 0.778 [0.753, 0.803] | 0.821 [0.803, 0.839] | 0.684 [0.665, 0.702] |
| CNN, attention pool | 0.686 [0.669, 0.704] | 0.677 [0.661, 0.694] | 0.654 [0.635, 0.675] | 0.718 [0.688, 0.748] | 0.801 [0.784, 0.818] | 0.720 [0.705, 0.735] |
| small ResNet, mean pool | 0.675 [0.657, 0.694] | 0.665 [0.647, 0.683] | 0.638 [0.615, 0.660] | 0.712 [0.685, 0.742] | 0.794 [0.777, 0.812] | 0.708 [0.689, 0.725] |
| small ResNet, attention pool | 0.686 [0.669, 0.703] | 0.663 [0.647, 0.680] | 0.602 [0.580, 0.624] | 0.770 [0.742, 0.797] | 0.819 [0.802, 0.838] | 0.694 [0.675, 0.712] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 237 | 128 | 237 |
| CNN, mean pool | 371 | 262 | 81 | 284 |
| CNN, attention pool | 414 | 219 | 103 | 262 |
| small ResNet, mean pool | 404 | 229 | 105 | 260 |
| small ResNet, attention pool | 381 | 252 | 84 | 281 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.437 | 0.419 | 0.237 | 0.637 | 0.438 | 0.308 |
| logistic regression | Cab Franc | 0.473 | 0.808 | 0.834 | 0.111 | 0.962 | 0.893 |
| logistic regression | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| logistic regression | Vidal | 0.515 | 0.644 | 0.882 | 0.148 | 0.683 | 0.770 |
| CNN, mean pool | Baco | 0.466 | 0.431 | 0.081 | 0.850 | 0.393 | 0.135 |
| CNN, mean pool | Cab Franc | 0.576 | 0.904 | 0.929 | 0.222 | 0.970 | 0.949 |
| CNN, mean pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| CNN, mean pool | Vidal | 0.548 | 0.624 | 0.763 | 0.333 | 0.705 | 0.733 |
| CNN, attention pool | Baco | 0.474 | 0.444 | 0.126 | 0.823 | 0.459 | 0.198 |
| CNN, attention pool | Cab Franc | 0.592 | 0.936 | 0.963 | 0.222 | 0.971 | 0.967 |
| CNN, attention pool | Chardonnay | 0.538 | 0.660 | 0.125 | 0.951 | 0.579 | 0.206 |
| CNN, attention pool | Vidal | 0.536 | 0.668 | 0.911 | 0.160 | 0.694 | 0.788 |
| small ResNet, mean pool | Baco | 0.500 | 0.476 | 0.222 | 0.779 | 0.545 | 0.316 |
| small ResNet, mean pool | Cab Franc | 0.518 | 0.896 | 0.925 | 0.111 | 0.965 | 0.945 |
| small ResNet, mean pool | Chardonnay | 0.554 | 0.664 | 0.182 | 0.926 | 0.571 | 0.276 |
| small ResNet, mean pool | Vidal | 0.529 | 0.624 | 0.799 | 0.259 | 0.692 | 0.742 |
| small ResNet, attention pool | Baco | 0.503 | 0.468 | 0.104 | 0.903 | 0.560 | 0.175 |
| small ResNet, attention pool | Cab Franc | 0.582 | 0.916 | 0.942 | 0.222 | 0.970 | 0.956 |
| small ResNet, attention pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet, attention pool | Vidal | 0.505 | 0.604 | 0.787 | 0.222 | 0.679 | 0.729 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | 0.581 [0.555, 0.607] | 0.602 [0.573, 0.631] | 0.509 [0.451, 0.562] | 0.807 [0.788, 0.827] | 0.690 [0.667, 0.712] |
| CNN, mean pool | 0.589 [0.564, 0.614] | 0.551 [0.531, 0.571] | 0.519 [0.496, 0.541] | 0.659 [0.615, 0.704] | 0.839 [0.821, 0.857] | 0.642 [0.621, 0.661] |
| CNN, attention pool | 0.602 [0.578, 0.626] | 0.554 [0.534, 0.574] | 0.514 [0.492, 0.539] | 0.690 [0.646, 0.735] | 0.850 [0.832, 0.868] | 0.641 [0.621, 0.661] |
| small ResNet, mean pool | 0.596 [0.571, 0.621] | 0.552 [0.531, 0.573] | 0.516 [0.491, 0.540] | 0.677 [0.633, 0.721] | 0.845 [0.826, 0.864] | 0.640 [0.619, 0.661] |
| small ResNet, attention pool | 0.613 [0.591, 0.635] | 0.650 [0.631, 0.669] | 0.681 [0.659, 0.702] | 0.544 [0.509, 0.580] | 0.836 [0.824, 0.848] | 0.751 [0.735, 0.766] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 465 | 307 | 111 | 115 |
| CNN, mean pool | 401 | 371 | 77 | 149 |
| CNN, attention pool | 397 | 375 | 70 | 156 |
| small ResNet, mean pool | 398 | 374 | 73 | 153 |
| small ResNet, attention pool | 526 | 246 | 103 | 123 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.431 | 0.480 | 0.524 | 0.339 | 0.717 | 0.606 |
| logistic regression | Cab Franc | 0.435 | 0.736 | 0.759 | 0.111 | 0.958 | 0.847 |
| logistic regression | Chardonnay | 0.491 | 0.436 | 0.294 | 0.689 | 0.627 | 0.400 |
| logistic regression | Vidal | 0.609 | 0.672 | 0.747 | 0.471 | 0.791 | 0.768 |
| CNN, mean pool | Baco | 0.463 | 0.359 | 0.265 | 0.661 | 0.714 | 0.386 |
| CNN, mean pool | Cab Franc | 0.530 | 0.816 | 0.838 | 0.222 | 0.967 | 0.898 |
| CNN, mean pool | Chardonnay | 0.531 | 0.456 | 0.263 | 0.800 | 0.700 | 0.382 |
| CNN, mean pool | Vidal | 0.559 | 0.572 | 0.588 | 0.529 | 0.770 | 0.667 |
| CNN, attention pool | Baco | 0.483 | 0.319 | 0.169 | 0.797 | 0.727 | 0.275 |
| CNN, attention pool | Cab Franc | 0.473 | 0.808 | 0.834 | 0.111 | 0.962 | 0.893 |
| CNN, attention pool | Chardonnay | 0.563 | 0.484 | 0.281 | 0.844 | 0.763 | 0.411 |
| CNN, attention pool | Vidal | 0.562 | 0.604 | 0.654 | 0.471 | 0.768 | 0.706 |
| small ResNet, mean pool | Baco | 0.481 | 0.476 | 0.471 | 0.492 | 0.748 | 0.578 |
| small ResNet, mean pool | Cab Franc | 0.532 | 0.820 | 0.842 | 0.222 | 0.967 | 0.900 |
| small ResNet, mean pool | Chardonnay | 0.530 | 0.408 | 0.094 | 0.967 | 0.833 | 0.169 |
| small ResNet, mean pool | Vidal | 0.507 | 0.504 | 0.500 | 0.515 | 0.734 | 0.595 |
| small ResNet, attention pool | Baco | 0.602 | 0.677 | 0.746 | 0.458 | 0.815 | 0.779 |
| small ResNet, attention pool | Cab Franc | 0.666 | 0.872 | 0.888 | 0.444 | 0.977 | 0.930 |
| small ResNet, attention pool | Chardonnay | 0.550 | 0.492 | 0.344 | 0.756 | 0.714 | 0.464 |
| small ResNet, attention pool | Vidal | 0.495 | 0.560 | 0.637 | 0.353 | 0.725 | 0.678 |
