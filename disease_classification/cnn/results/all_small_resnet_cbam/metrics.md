# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| small ResNet + CBAM, mean pool | 0.610 [0.591, 0.629] | 0.594 [0.575, 0.612] | 0.663 [0.635, 0.690] | 0.558 [0.535, 0.582] | 0.437 [0.420, 0.454] | 0.526 [0.507, 0.546] |
| small ResNet + CBAM, attention pool | 0.583 [0.565, 0.601] | 0.564 [0.548, 0.582] | 0.643 [0.616, 0.671] | 0.523 [0.503, 0.544] | 0.411 [0.396, 0.426] | 0.502 [0.482, 0.519] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| small ResNet + CBAM, mean pool | 169 | 86 | 218 | 275 |
| small ResNet + CBAM, attention pool | 164 | 91 | 235 | 258 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| small ResNet + CBAM, mean pool | Baco | 0.527 | 0.492 | 0.785 | 0.270 | 0.449 | 0.571 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.488 | 0.484 | 0.672 | 0.305 | 0.480 | 0.560 |
| small ResNet + CBAM, mean pool | Vidal | 0.500 | 0.804 | 0.115 | 0.884 | 0.103 | 0.109 |
| small ResNet + CBAM, attention pool | Baco | 0.501 | 0.476 | 0.682 | 0.319 | 0.432 | 0.529 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.431 | 0.424 | 0.713 | 0.148 | 0.444 | 0.547 |
| small ResNet + CBAM, attention pool | Vidal | 0.510 | 0.792 | 0.154 | 0.866 | 0.118 | 0.133 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| small ResNet + CBAM, mean pool | 0.686 [0.666, 0.704] | 0.655 [0.636, 0.672] | 0.576 [0.553, 0.597] | 0.795 [0.767, 0.825] | 0.832 [0.812, 0.853] | 0.681 [0.660, 0.699] |
| small ResNet + CBAM, attention pool | 0.678 [0.660, 0.695] | 0.684 [0.666, 0.700] | 0.702 [0.678, 0.724] | 0.654 [0.626, 0.681] | 0.781 [0.767, 0.796] | 0.739 [0.723, 0.755] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| small ResNet + CBAM, mean pool | 367 | 270 | 74 | 287 |
| small ResNet + CBAM, attention pool | 447 | 190 | 125 | 236 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| small ResNet + CBAM, mean pool | Baco | 0.482 | 0.448 | 0.089 | 0.876 | 0.462 | 0.149 |
| small ResNet + CBAM, mean pool | Cab Franc | 0.572 | 0.896 | 0.921 | 0.222 | 0.969 | 0.945 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet + CBAM, mean pool | Vidal | 0.539 | 0.612 | 0.728 | 0.351 | 0.716 | 0.722 |
| small ResNet + CBAM, attention pool | Baco | 0.514 | 0.504 | 0.400 | 0.628 | 0.562 | 0.468 |
| small ResNet + CBAM, attention pool | Cab Franc | 0.533 | 0.924 | 0.954 | 0.111 | 0.966 | 0.960 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.546 | 0.664 | 0.148 | 0.944 | 0.591 | 0.236 |
| small ResNet + CBAM, attention pool | Vidal | 0.505 | 0.644 | 0.867 | 0.143 | 0.694 | 0.771 |
