# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| CNN + ECA, mean pool | 0.603 [0.579, 0.625] | 0.591 [0.570, 0.612] | 0.639 [0.600, 0.678] | 0.566 [0.542, 0.590] | 0.432 [0.412, 0.453] | 0.516 [0.490, 0.541] |
| CNN + ECA, attention pool | 0.590 [0.568, 0.611] | 0.603 [0.582, 0.623] | 0.549 [0.514, 0.584] | 0.631 [0.604, 0.657] | 0.435 [0.411, 0.458] | 0.485 [0.459, 0.510] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| CNN + ECA, mean pool | 163 | 92 | 214 | 279 |
| CNN + ECA, attention pool | 140 | 115 | 182 | 311 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| CNN + ECA, mean pool | Baco | 0.480 | 0.448 | 0.720 | 0.241 | 0.418 | 0.529 |
| CNN + ECA, mean pool | Chardonnay | 0.477 | 0.472 | 0.697 | 0.258 | 0.472 | 0.563 |
| CNN + ECA, mean pool | Vidal | 0.492 | 0.852 | 0.038 | 0.946 | 0.077 | 0.051 |
| CNN + ECA, attention pool | Baco | 0.478 | 0.464 | 0.579 | 0.376 | 0.413 | 0.482 |
| CNN + ECA, attention pool | Chardonnay | 0.503 | 0.500 | 0.631 | 0.375 | 0.490 | 0.552 |
| CNN + ECA, attention pool | Vidal | 0.488 | 0.844 | 0.038 | 0.938 | 0.067 | 0.049 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| CNN + ECA, mean pool | 0.686 [0.669, 0.702] | 0.685 [0.669, 0.701] | 0.684 [0.664, 0.703] | 0.687 [0.659, 0.715] | 0.794 [0.779, 0.810] | 0.735 [0.721, 0.749] |
| CNN + ECA, attention pool | 0.681 [0.663, 0.698] | 0.667 [0.650, 0.684] | 0.633 [0.612, 0.653] | 0.729 [0.704, 0.756] | 0.804 [0.789, 0.821] | 0.708 [0.691, 0.725] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| CNN + ECA, mean pool | 436 | 201 | 113 | 248 |
| CNN + ECA, attention pool | 403 | 234 | 98 | 263 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| CNN + ECA, mean pool | Baco | 0.489 | 0.460 | 0.156 | 0.823 | 0.512 | 0.239 |
| CNN + ECA, mean pool | Cab Franc | 0.545 | 0.948 | 0.979 | 0.111 | 0.967 | 0.973 |
| CNN + ECA, mean pool | Chardonnay | 0.569 | 0.664 | 0.250 | 0.889 | 0.550 | 0.344 |
| CNN + ECA, mean pool | Vidal | 0.519 | 0.668 | 0.908 | 0.130 | 0.701 | 0.791 |
| CNN + ECA, attention pool | Baco | 0.483 | 0.456 | 0.178 | 0.788 | 0.500 | 0.262 |
| CNN + ECA, attention pool | Cab Franc | 0.586 | 0.924 | 0.950 | 0.222 | 0.970 | 0.960 |
| CNN + ECA, attention pool | Chardonnay | 0.539 | 0.668 | 0.102 | 0.975 | 0.692 | 0.178 |
| CNN + ECA, attention pool | Vidal | 0.498 | 0.620 | 0.815 | 0.182 | 0.691 | 0.748 |
