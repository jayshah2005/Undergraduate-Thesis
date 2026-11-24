# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | 0.607 [0.582, 0.632] | 0.557 [0.510, 0.600] | 0.633 [0.602, 0.665] | 0.440 [0.411, 0.468] | 0.491 [0.459, 0.523] |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | 0.567 [0.539, 0.594] | 0.557 [0.510, 0.600] | 0.572 [0.540, 0.602] | 0.402 [0.374, 0.429] | 0.467 [0.433, 0.499] |
| small ResNet, mean ∥ max pool | 0.577 [0.553, 0.601] | 0.553 [0.529, 0.579] | 0.651 [0.612, 0.686] | 0.503 [0.471, 0.535] | 0.404 [0.384, 0.425] | 0.498 [0.474, 0.523] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| small ResNet, mean pool | 142 | 113 | 181 | 312 |
| small ResNet, attention pool | 142 | 113 | 211 | 282 |
| small ResNet, mean ∥ max pool | 166 | 89 | 245 | 248 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| small ResNet, mean pool | Baco | 0.524 | 0.532 | 0.467 | 0.582 | 0.459 | 0.463 |
| small ResNet, mean pool | Chardonnay | 0.525 | 0.520 | 0.746 | 0.305 | 0.506 | 0.603 |
| small ResNet, mean pool | Vidal | 0.446 | 0.768 | 0.038 | 0.853 | 0.029 | 0.033 |
| small ResNet, attention pool | Baco | 0.477 | 0.484 | 0.430 | 0.525 | 0.407 | 0.418 |
| small ResNet, attention pool | Chardonnay | 0.440 | 0.432 | 0.762 | 0.117 | 0.451 | 0.567 |
| small ResNet, attention pool | Vidal | 0.488 | 0.784 | 0.115 | 0.862 | 0.088 | 0.100 |
| small ResNet, mean ∥ max pool | Baco | 0.509 | 0.492 | 0.636 | 0.383 | 0.439 | 0.519 |
| small ResNet, mean ∥ max pool | Chardonnay | 0.486 | 0.480 | 0.754 | 0.219 | 0.479 | 0.586 |
| small ResNet, mean ∥ max pool | Vidal | 0.486 | 0.688 | 0.231 | 0.741 | 0.094 | 0.133 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| small ResNet, mean pool | 0.659 [0.641, 0.678] | 0.630 [0.612, 0.648] | 0.554 [0.529, 0.578] | 0.765 [0.737, 0.792] | 0.806 [0.786, 0.826] | 0.657 [0.636, 0.676] |
| small ResNet, attention pool | 0.668 [0.646, 0.688] | 0.646 [0.625, 0.666] | 0.590 [0.562, 0.617] | 0.745 [0.715, 0.776] | 0.803 [0.782, 0.824] | 0.681 [0.657, 0.702] |
| small ResNet, mean ∥ max pool | 0.666 [0.647, 0.685] | 0.631 [0.611, 0.650] | 0.540 [0.515, 0.564] | 0.792 [0.765, 0.820] | 0.821 [0.800, 0.843] | 0.652 [0.630, 0.672] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| small ResNet, mean pool | 353 | 284 | 85 | 276 |
| small ResNet, attention pool | 376 | 261 | 92 | 269 |
| small ResNet, mean ∥ max pool | 344 | 293 | 75 | 286 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| small ResNet, mean pool | Baco | 0.497 | 0.476 | 0.259 | 0.735 | 0.538 | 0.350 |
| small ResNet, mean pool | Cab Franc | 0.502 | 0.864 | 0.892 | 0.111 | 0.964 | 0.927 |
| small ResNet, mean pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet, mean pool | Vidal | 0.492 | 0.516 | 0.555 | 0.429 | 0.686 | 0.613 |
| small ResNet, attention pool | Baco | 0.508 | 0.488 | 0.281 | 0.735 | 0.559 | 0.374 |
| small ResNet, attention pool | Cab Franc | 0.485 | 0.832 | 0.859 | 0.111 | 0.963 | 0.908 |
| small ResNet, attention pool | Chardonnay | 0.542 | 0.672 | 0.102 | 0.981 | 0.750 | 0.180 |
| small ResNet, attention pool | Vidal | 0.521 | 0.592 | 0.705 | 0.338 | 0.705 | 0.705 |
| small ResNet, mean ∥ max pool | Baco | 0.503 | 0.480 | 0.237 | 0.770 | 0.552 | 0.332 |
| small ResNet, mean ∥ max pool | Cab Franc | 0.497 | 0.856 | 0.884 | 0.111 | 0.964 | 0.922 |
| small ResNet, mean ∥ max pool | Chardonnay | 0.517 | 0.656 | 0.045 | 0.988 | 0.667 | 0.085 |
| small ResNet, mean ∥ max pool | Vidal | 0.521 | 0.532 | 0.549 | 0.494 | 0.709 | 0.619 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | 0.546 [0.523, 0.571] | 0.517 [0.491, 0.544] | 0.649 [0.599, 0.694] | 0.837 [0.817, 0.857] | 0.639 [0.616, 0.663] |
| small ResNet, mean pool | 0.597 [0.574, 0.620] | 0.521 [0.502, 0.541] | 0.460 [0.437, 0.483] | 0.734 [0.698, 0.770] | 0.858 [0.840, 0.878] | 0.599 [0.578, 0.620] |
| small ResNet, attention pool | 0.598 [0.577, 0.619] | 0.512 [0.494, 0.530] | 0.443 [0.423, 0.463] | 0.752 [0.721, 0.788] | 0.862 [0.845, 0.880] | 0.586 [0.566, 0.604] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 401 | 375 | 78 | 144 |
| small ResNet, mean pool | 357 | 419 | 59 | 163 |
| small ResNet, attention pool | 344 | 432 | 55 | 167 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.440 | 0.387 | 0.339 | 0.542 | 0.703 | 0.457 |
| logistic regression | Cab Franc | 0.497 | 0.752 | 0.772 | 0.222 | 0.964 | 0.857 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.532 | 0.624 | 0.720 | 0.344 | 0.761 | 0.740 |
| small ResNet, mean pool | Baco | 0.508 | 0.375 | 0.254 | 0.763 | 0.774 | 0.382 |
| small ResNet, mean pool | Cab Franc | 0.491 | 0.844 | 0.871 | 0.111 | 0.963 | 0.915 |
| small ResNet, mean pool | Chardonnay | 0.497 | 0.384 | 0.094 | 0.900 | 0.625 | 0.163 |
| small ResNet, mean pool | Vidal | 0.507 | 0.480 | 0.452 | 0.562 | 0.750 | 0.564 |
| small ResNet, attention pool | Baco | 0.492 | 0.351 | 0.222 | 0.763 | 0.750 | 0.343 |
| small ResNet, attention pool | Cab Franc | 0.470 | 0.804 | 0.830 | 0.111 | 0.962 | 0.891 |
| small ResNet, attention pool | Chardonnay | 0.505 | 0.376 | 0.044 | 0.967 | 0.700 | 0.082 |
| small ResNet, attention pool | Vidal | 0.521 | 0.516 | 0.511 | 0.531 | 0.760 | 0.611 |
