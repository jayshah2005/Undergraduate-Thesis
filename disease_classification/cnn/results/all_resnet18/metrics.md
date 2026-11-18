# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | 0.584 [0.559, 0.610] | 0.678 [0.635, 0.722] | 0.535 [0.505, 0.566] | 0.430 [0.408, 0.453] | 0.527 [0.499, 0.554] |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | 0.557 [0.532, 0.583] | 0.624 [0.576, 0.671] | 0.523 [0.493, 0.552] | 0.404 [0.380, 0.428] | 0.490 [0.461, 0.519] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| ResNet-18, mean pool | 173 | 82 | 229 | 264 |
| ResNet-18, attention pool | 159 | 96 | 235 | 258 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| ResNet-18, mean pool | Baco | 0.548 | 0.524 | 0.720 | 0.376 | 0.467 | 0.566 |
| ResNet-18, mean pool | Chardonnay | 0.502 | 0.496 | 0.746 | 0.258 | 0.489 | 0.591 |
| ResNet-18, mean pool | Vidal | 0.493 | 0.732 | 0.192 | 0.795 | 0.098 | 0.130 |
| ResNet-18, attention pool | Baco | 0.507 | 0.492 | 0.617 | 0.397 | 0.437 | 0.512 |
| ResNet-18, attention pool | Chardonnay | 0.419 | 0.412 | 0.721 | 0.117 | 0.438 | 0.545 |
| ResNet-18, attention pool | Vidal | 0.514 | 0.768 | 0.192 | 0.835 | 0.119 | 0.147 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| ResNet-18, mean pool | 0.672 [0.647, 0.693] | 0.649 [0.627, 0.669] | 0.590 [0.565, 0.615] | 0.753 [0.712, 0.789] | 0.809 [0.783, 0.832] | 0.682 [0.661, 0.703] |
| ResNet-18, attention pool | 0.667 [0.643, 0.689] | 0.642 [0.620, 0.662] | 0.578 [0.554, 0.603] | 0.756 [0.715, 0.792] | 0.807 [0.781, 0.832] | 0.673 [0.653, 0.694] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| ResNet-18, mean pool | 376 | 261 | 89 | 272 |
| ResNet-18, attention pool | 368 | 269 | 88 | 273 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| ResNet-18, mean pool | Baco | 0.460 | 0.431 | 0.133 | 0.788 | 0.429 | 0.203 |
| ResNet-18, mean pool | Cab Franc | 0.701 | 0.940 | 0.959 | 0.444 | 0.979 | 0.969 |
| ResNet-18, mean pool | Chardonnay | 0.545 | 0.652 | 0.182 | 0.907 | 0.516 | 0.269 |
| ResNet-18, mean pool | Vidal | 0.529 | 0.572 | 0.642 | 0.416 | 0.712 | 0.675 |
| ResNet-18, attention pool | Baco | 0.472 | 0.440 | 0.104 | 0.841 | 0.438 | 0.168 |
| ResNet-18, attention pool | Cab Franc | 0.537 | 0.932 | 0.963 | 0.111 | 0.967 | 0.965 |
| ResNet-18, attention pool | Chardonnay | 0.541 | 0.644 | 0.193 | 0.889 | 0.486 | 0.276 |
| ResNet-18, attention pool | Vidal | 0.518 | 0.552 | 0.607 | 0.429 | 0.705 | 0.652 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | 0.587 [0.561, 0.613] | 0.612 [0.584, 0.640] | 0.500 [0.441, 0.554] | 0.811 [0.791, 0.830] | 0.698 [0.675, 0.719] |
| ResNet-18, mean pool | 0.602 [0.575, 0.630] | 0.584 [0.560, 0.608] | 0.570 [0.543, 0.598] | 0.635 [0.586, 0.685] | 0.845 [0.827, 0.863] | 0.681 [0.657, 0.703] |
| ResNet-18, attention pool | 0.621 [0.594, 0.646] | 0.588 [0.567, 0.610] | 0.562 [0.537, 0.588] | 0.680 [0.631, 0.725] | 0.860 [0.841, 0.878] | 0.680 [0.658, 0.700] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 475 | 301 | 111 | 111 |
| ResNet-18, mean pool | 442 | 334 | 81 | 141 |
| ResNet-18, attention pool | 436 | 340 | 71 | 151 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.448 | 0.496 | 0.540 | 0.356 | 0.729 | 0.620 |
| logistic regression | Cab Franc | 0.444 | 0.752 | 0.776 | 0.111 | 0.959 | 0.858 |
| logistic regression | Chardonnay | 0.481 | 0.420 | 0.263 | 0.700 | 0.609 | 0.367 |
| logistic regression | Vidal | 0.590 | 0.680 | 0.774 | 0.406 | 0.791 | 0.783 |
| ResNet-18, mean pool | Baco | 0.467 | 0.427 | 0.392 | 0.542 | 0.733 | 0.510 |
| ResNet-18, mean pool | Cab Franc | 0.555 | 0.864 | 0.888 | 0.222 | 0.968 | 0.926 |
| ResNet-18, mean pool | Chardonnay | 0.575 | 0.524 | 0.394 | 0.756 | 0.741 | 0.514 |
| ResNet-18, mean pool | Vidal | 0.549 | 0.520 | 0.489 | 0.609 | 0.784 | 0.603 |
| ResNet-18, attention pool | Baco | 0.554 | 0.516 | 0.481 | 0.627 | 0.805 | 0.603 |
| ResNet-18, attention pool | Cab Franc | 0.543 | 0.840 | 0.863 | 0.222 | 0.967 | 0.912 |
| ResNet-18, attention pool | Chardonnay | 0.561 | 0.500 | 0.344 | 0.778 | 0.733 | 0.468 |
| ResNet-18, attention pool | Vidal | 0.549 | 0.496 | 0.441 | 0.656 | 0.788 | 0.566 |
