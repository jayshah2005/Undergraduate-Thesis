# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| ResNet-18, mean pool | 0.601 [0.572, 0.627] | 0.582 [0.555, 0.607] | 0.661 [0.614, 0.705] | 0.540 [0.508, 0.571] | 0.425 [0.400, 0.448] | 0.518 [0.488, 0.545] |
| ResNet-18, attention pool | 0.577 [0.549, 0.603] | 0.556 [0.531, 0.582] | 0.642 [0.598, 0.685] | 0.512 [0.482, 0.543] | 0.403 [0.380, 0.425] | 0.495 [0.467, 0.521] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| ResNet-18, mean pool | 168 | 86 | 227 | 267 |
| ResNet-18, attention pool | 163 | 91 | 241 | 253 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| ResNet-18, mean pool | Baco | 0.507 | 0.488 | 0.645 | 0.369 | 0.437 | 0.521 |
| ResNet-18, mean pool | Chardonnay | 0.494 | 0.488 | 0.762 | 0.227 | 0.484 | 0.592 |
| ResNet-18, mean pool | Vidal | 0.533 | 0.768 | 0.240 | 0.827 | 0.133 | 0.171 |
| ResNet-18, attention pool | Baco | 0.522 | 0.504 | 0.654 | 0.390 | 0.449 | 0.532 |
| ResNet-18, attention pool | Chardonnay | 0.443 | 0.436 | 0.730 | 0.156 | 0.452 | 0.558 |
| ResNet-18, attention pool | Vidal | 0.476 | 0.728 | 0.160 | 0.791 | 0.078 | 0.105 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | 0.634 [0.614, 0.655] | 0.626 [0.599, 0.652] | 0.649 [0.614, 0.682] | 0.756 [0.737, 0.775] | 0.685 [0.664, 0.704] |
| ResNet-18, mean pool | 0.670 [0.649, 0.692] | 0.639 [0.619, 0.659] | 0.555 [0.529, 0.578] | 0.786 [0.751, 0.822] | 0.818 [0.794, 0.844] | 0.661 [0.640, 0.682] |
| ResNet-18, attention pool | 0.669 [0.649, 0.689] | 0.637 [0.618, 0.656] | 0.551 [0.528, 0.575] | 0.786 [0.756, 0.816] | 0.817 [0.795, 0.840] | 0.658 [0.637, 0.678] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 237 | 128 | 237 |
| ResNet-18, mean pool | 351 | 282 | 78 | 287 |
| ResNet-18, attention pool | 349 | 284 | 78 | 287 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.437 | 0.419 | 0.237 | 0.637 | 0.438 | 0.308 |
| logistic regression | Cab Franc | 0.473 | 0.808 | 0.834 | 0.111 | 0.962 | 0.893 |
| logistic regression | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| logistic regression | Vidal | 0.515 | 0.644 | 0.882 | 0.148 | 0.683 | 0.770 |
| ResNet-18, mean pool | Baco | 0.479 | 0.444 | 0.081 | 0.876 | 0.440 | 0.138 |
| ResNet-18, mean pool | Cab Franc | 0.590 | 0.932 | 0.959 | 0.222 | 0.971 | 0.965 |
| ResNet-18, mean pool | Chardonnay | 0.550 | 0.652 | 0.205 | 0.895 | 0.514 | 0.293 |
| ResNet-18, mean pool | Vidal | 0.522 | 0.528 | 0.538 | 0.506 | 0.695 | 0.607 |
| ResNet-18, attention pool | Baco | 0.471 | 0.440 | 0.119 | 0.823 | 0.444 | 0.187 |
| ResNet-18, attention pool | Cab Franc | 0.642 | 0.928 | 0.950 | 0.333 | 0.974 | 0.962 |
| ResNet-18, attention pool | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| ResNet-18, attention pool | Vidal | 0.516 | 0.524 | 0.538 | 0.494 | 0.689 | 0.605 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | 0.581 [0.555, 0.607] | 0.602 [0.573, 0.631] | 0.509 [0.451, 0.562] | 0.807 [0.788, 0.827] | 0.690 [0.667, 0.712] |
| ResNet-18, mean pool | 0.577 [0.553, 0.599] | 0.565 [0.545, 0.585] | 0.556 [0.532, 0.579] | 0.597 [0.553, 0.637] | 0.825 [0.809, 0.840] | 0.664 [0.645, 0.683] |
| ResNet-18, attention pool | 0.604 [0.578, 0.630] | 0.586 [0.563, 0.608] | 0.571 [0.544, 0.598] | 0.637 [0.593, 0.681] | 0.843 [0.825, 0.860] | 0.681 [0.658, 0.702] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 465 | 307 | 111 | 115 |
| ResNet-18, mean pool | 429 | 343 | 91 | 135 |
| ResNet-18, attention pool | 441 | 331 | 82 | 144 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.431 | 0.480 | 0.524 | 0.339 | 0.717 | 0.606 |
| logistic regression | Cab Franc | 0.435 | 0.736 | 0.759 | 0.111 | 0.958 | 0.847 |
| logistic regression | Chardonnay | 0.491 | 0.436 | 0.294 | 0.689 | 0.627 | 0.400 |
| logistic regression | Vidal | 0.609 | 0.672 | 0.747 | 0.471 | 0.791 | 0.768 |
| ResNet-18, mean pool | Baco | 0.484 | 0.427 | 0.376 | 0.593 | 0.747 | 0.500 |
| ResNet-18, mean pool | Cab Franc | 0.510 | 0.880 | 0.909 | 0.111 | 0.965 | 0.936 |
| ResNet-18, mean pool | Chardonnay | 0.515 | 0.460 | 0.319 | 0.711 | 0.662 | 0.430 |
| ResNet-18, mean pool | Vidal | 0.499 | 0.492 | 0.484 | 0.515 | 0.727 | 0.581 |
| ResNet-18, attention pool | Baco | 0.509 | 0.456 | 0.407 | 0.610 | 0.770 | 0.533 |
| ResNet-18, attention pool | Cab Franc | 0.613 | 0.872 | 0.892 | 0.333 | 0.973 | 0.931 |
| ResNet-18, attention pool | Chardonnay | 0.545 | 0.480 | 0.312 | 0.778 | 0.714 | 0.435 |
| ResNet-18, attention pool | Vidal | 0.529 | 0.536 | 0.544 | 0.515 | 0.750 | 0.631 |
