# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | 0.549 [0.529, 0.570] | 0.698 [0.667, 0.729] | 0.473 [0.448, 0.497] | 0.406 [0.390, 0.422] | 0.514 [0.493, 0.533] |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | 0.572 [0.549, 0.595] | 0.737 [0.698, 0.776] | 0.487 [0.460, 0.513] | 0.426 [0.408, 0.445] | 0.540 [0.516, 0.564] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| small ResNet, mean pool | 178 | 77 | 260 | 233 |
| small ResNet, attention pool | 188 | 67 | 253 | 240 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| small ResNet, mean pool | Baco | 0.503 | 0.460 | 0.822 | 0.184 | 0.433 | 0.568 |
| small ResNet, mean pool | Chardonnay | 0.442 | 0.436 | 0.697 | 0.188 | 0.450 | 0.547 |
| small ResNet, mean pool | Vidal | 0.505 | 0.752 | 0.192 | 0.817 | 0.109 | 0.139 |
| small ResNet, attention pool | Baco | 0.518 | 0.488 | 0.738 | 0.298 | 0.444 | 0.554 |
| small ResNet, attention pool | Chardonnay | 0.520 | 0.512 | 0.844 | 0.195 | 0.500 | 0.628 |
| small ResNet, attention pool | Vidal | 0.502 | 0.716 | 0.231 | 0.772 | 0.105 | 0.145 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | 0.660 [0.641, 0.679] | 0.603 [0.578, 0.626] | 0.762 [0.731, 0.792] | 0.817 [0.797, 0.837] | 0.694 [0.673, 0.713] |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | 0.664 [0.646, 0.681] | 0.608 [0.584, 0.630] | 0.765 [0.737, 0.792] | 0.820 [0.802, 0.838] | 0.698 [0.680, 0.715] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| small ResNet, mean pool | 384 | 253 | 86 | 275 |
| small ResNet, attention pool | 387 | 250 | 85 | 276 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| small ResNet, mean pool | Baco | 0.504 | 0.476 | 0.185 | 0.823 | 0.556 | 0.278 |
| small ResNet, mean pool | Cab Franc | 0.512 | 0.884 | 0.913 | 0.111 | 0.965 | 0.938 |
| small ResNet, mean pool | Chardonnay | 0.555 | 0.672 | 0.159 | 0.951 | 0.636 | 0.255 |
| small ResNet, mean pool | Vidal | 0.537 | 0.608 | 0.723 | 0.351 | 0.714 | 0.718 |
| small ResNet, attention pool | Baco | 0.474 | 0.440 | 0.081 | 0.867 | 0.423 | 0.137 |
| small ResNet, attention pool | Cab Franc | 0.524 | 0.908 | 0.938 | 0.111 | 0.966 | 0.952 |
| small ResNet, attention pool | Chardonnay | 0.538 | 0.664 | 0.114 | 0.963 | 0.625 | 0.192 |
| small ResNet, attention pool | Vidal | 0.541 | 0.644 | 0.809 | 0.273 | 0.714 | 0.759 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | 0.587 [0.561, 0.613] | 0.612 [0.584, 0.640] | 0.500 [0.441, 0.554] | 0.811 [0.791, 0.830] | 0.698 [0.675, 0.719] |
| small ResNet, mean pool | 0.637 [0.613, 0.658] | 0.565 [0.545, 0.585] | 0.508 [0.485, 0.531] | 0.766 [0.725, 0.806] | 0.883 [0.865, 0.901] | 0.645 [0.624, 0.666] |
| small ResNet, attention pool | 0.627 [0.601, 0.650] | 0.609 [0.588, 0.629] | 0.595 [0.571, 0.619] | 0.658 [0.617, 0.698] | 0.859 [0.843, 0.874] | 0.703 [0.684, 0.722] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 475 | 301 | 111 | 111 |
| small ResNet, mean pool | 394 | 382 | 52 | 170 |
| small ResNet, attention pool | 462 | 314 | 76 | 146 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.448 | 0.496 | 0.540 | 0.356 | 0.729 | 0.620 |
| logistic regression | Cab Franc | 0.444 | 0.752 | 0.776 | 0.111 | 0.959 | 0.858 |
| logistic regression | Chardonnay | 0.481 | 0.420 | 0.263 | 0.700 | 0.609 | 0.367 |
| logistic regression | Vidal | 0.590 | 0.680 | 0.774 | 0.406 | 0.791 | 0.783 |
| small ResNet, mean pool | Baco | 0.513 | 0.347 | 0.196 | 0.831 | 0.787 | 0.314 |
| small ResNet, mean pool | Cab Franc | 0.557 | 0.868 | 0.892 | 0.222 | 0.968 | 0.929 |
| small ResNet, mean pool | Chardonnay | 0.563 | 0.456 | 0.181 | 0.944 | 0.853 | 0.299 |
| small ResNet, mean pool | Vidal | 0.569 | 0.588 | 0.608 | 0.531 | 0.790 | 0.687 |
| small ResNet, attention pool | Baco | 0.593 | 0.540 | 0.492 | 0.695 | 0.838 | 0.620 |
| small ResNet, attention pool | Cab Franc | 0.586 | 0.820 | 0.838 | 0.333 | 0.971 | 0.900 |
| small ResNet, attention pool | Chardonnay | 0.553 | 0.472 | 0.263 | 0.844 | 0.750 | 0.389 |
| small ResNet, attention pool | Vidal | 0.539 | 0.604 | 0.672 | 0.406 | 0.767 | 0.716 |
