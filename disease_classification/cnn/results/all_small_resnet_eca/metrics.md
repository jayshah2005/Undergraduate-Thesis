# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| small ResNet + ECA, mean pool | 0.607 [0.589, 0.625] | 0.596 [0.578, 0.615] | 0.639 [0.612, 0.667] | 0.574 [0.550, 0.600] | 0.437 [0.420, 0.455] | 0.519 [0.500, 0.538] |
| small ResNet + ECA, attention pool | 0.580 [0.560, 0.599] | 0.553 [0.533, 0.574] | 0.663 [0.631, 0.694] | 0.497 [0.473, 0.521] | 0.405 [0.388, 0.422] | 0.503 [0.482, 0.522] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| small ResNet + ECA, mean pool | 163 | 92 | 210 | 283 |
| small ResNet + ECA, attention pool | 169 | 86 | 248 | 245 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| small ResNet + ECA, mean pool | Baco | 0.527 | 0.504 | 0.692 | 0.362 | 0.451 | 0.546 |
| small ResNet + ECA, mean pool | Chardonnay | 0.485 | 0.480 | 0.713 | 0.258 | 0.478 | 0.572 |
| small ResNet + ECA, mean pool | Vidal | 0.483 | 0.804 | 0.077 | 0.888 | 0.074 | 0.075 |
| small ResNet + ECA, attention pool | Baco | 0.489 | 0.456 | 0.729 | 0.248 | 0.424 | 0.536 |
| small ResNet + ECA, attention pool | Chardonnay | 0.450 | 0.444 | 0.697 | 0.203 | 0.455 | 0.550 |
| small ResNet + ECA, attention pool | Vidal | 0.526 | 0.760 | 0.231 | 0.821 | 0.130 | 0.167 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| small ResNet + ECA, mean pool | 0.683 [0.666, 0.701] | 0.662 [0.644, 0.680] | 0.608 [0.586, 0.628] | 0.759 [0.731, 0.787] | 0.816 [0.798, 0.836] | 0.697 [0.679, 0.714] |
| small ResNet + ECA, attention pool | 0.680 [0.662, 0.699] | 0.680 [0.662, 0.698] | 0.681 [0.659, 0.703] | 0.679 [0.648, 0.709] | 0.789 [0.773, 0.806] | 0.731 [0.715, 0.748] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| small ResNet + ECA, mean pool | 387 | 250 | 87 | 274 |
| small ResNet + ECA, attention pool | 434 | 203 | 116 | 245 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| small ResNet + ECA, mean pool | Baco | 0.463 | 0.427 | 0.067 | 0.858 | 0.360 | 0.113 |
| small ResNet + ECA, mean pool | Cab Franc | 0.533 | 0.924 | 0.954 | 0.111 | 0.966 | 0.960 |
| small ResNet + ECA, mean pool | Chardonnay | 0.547 | 0.668 | 0.136 | 0.957 | 0.632 | 0.224 |
| small ResNet + ECA, mean pool | Vidal | 0.529 | 0.628 | 0.786 | 0.273 | 0.708 | 0.745 |
| small ResNet + ECA, attention pool | Baco | 0.482 | 0.460 | 0.230 | 0.735 | 0.508 | 0.316 |
| small ResNet + ECA, attention pool | Cab Franc | 0.545 | 0.948 | 0.979 | 0.111 | 0.967 | 0.973 |
| small ResNet + ECA, attention pool | Chardonnay | 0.565 | 0.668 | 0.216 | 0.914 | 0.576 | 0.314 |
| small ResNet + ECA, attention pool | Vidal | 0.512 | 0.644 | 0.855 | 0.169 | 0.698 | 0.769 |
