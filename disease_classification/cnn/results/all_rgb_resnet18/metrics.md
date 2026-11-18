# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | 0.584 [0.557, 0.611] | 0.592 [0.549, 0.635] | 0.580 [0.548, 0.613] | 0.422 [0.396, 0.448] | 0.493 [0.462, 0.523] |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | 0.594 [0.567, 0.620] | 0.663 [0.620, 0.706] | 0.558 [0.525, 0.590] | 0.437 [0.412, 0.461] | 0.526 [0.497, 0.554] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| ResNet-18, mean pool | 151 | 104 | 207 | 286 |
| ResNet-18, attention pool | 169 | 86 | 218 | 275 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| ResNet-18, mean pool | Baco | 0.523 | 0.516 | 0.570 | 0.475 | 0.452 | 0.504 |
| ResNet-18, mean pool | Chardonnay | 0.509 | 0.504 | 0.713 | 0.305 | 0.494 | 0.584 |
| ResNet-18, mean pool | Vidal | 0.459 | 0.732 | 0.115 | 0.804 | 0.064 | 0.082 |
| ResNet-18, attention pool | Baco | 0.549 | 0.536 | 0.645 | 0.454 | 0.473 | 0.545 |
| ResNet-18, attention pool | Chardonnay | 0.471 | 0.464 | 0.779 | 0.164 | 0.470 | 0.586 |
| ResNet-18, attention pool | Vidal | 0.520 | 0.780 | 0.192 | 0.848 | 0.128 | 0.154 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| ResNet-18, mean pool | 0.665 [0.643, 0.685] | 0.639 [0.618, 0.659] | 0.573 [0.545, 0.598] | 0.756 [0.726, 0.787] | 0.806 [0.784, 0.827] | 0.670 [0.647, 0.691] |
| ResNet-18, attention pool | 0.675 [0.655, 0.694] | 0.647 [0.628, 0.666] | 0.575 [0.549, 0.598] | 0.776 [0.745, 0.806] | 0.819 [0.798, 0.839] | 0.675 [0.655, 0.695] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| ResNet-18, mean pool | 365 | 272 | 88 | 273 |
| ResNet-18, attention pool | 366 | 271 | 81 | 280 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| ResNet-18, mean pool | Baco | 0.480 | 0.456 | 0.207 | 0.752 | 0.500 | 0.293 |
| ResNet-18, mean pool | Cab Franc | 0.516 | 0.892 | 0.921 | 0.111 | 0.965 | 0.943 |
| ResNet-18, mean pool | Chardonnay | 0.537 | 0.656 | 0.136 | 0.938 | 0.545 | 0.218 |
| ResNet-18, mean pool | Vidal | 0.525 | 0.552 | 0.595 | 0.455 | 0.710 | 0.648 |
| ResNet-18, attention pool | Baco | 0.520 | 0.496 | 0.252 | 0.788 | 0.586 | 0.352 |
| ResNet-18, attention pool | Cab Franc | 0.527 | 0.912 | 0.942 | 0.111 | 0.966 | 0.954 |
| ResNet-18, attention pool | Chardonnay | 0.564 | 0.680 | 0.170 | 0.957 | 0.682 | 0.273 |
| ResNet-18, attention pool | Vidal | 0.487 | 0.500 | 0.520 | 0.455 | 0.682 | 0.590 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | 0.546 [0.523, 0.571] | 0.517 [0.491, 0.544] | 0.649 [0.599, 0.694] | 0.837 [0.817, 0.857] | 0.639 [0.616, 0.663] |
| ResNet-18, mean pool | 0.640 [0.617, 0.662] | 0.583 [0.563, 0.602] | 0.537 [0.514, 0.559] | 0.743 [0.703, 0.784] | 0.880 [0.863, 0.896] | 0.667 [0.647, 0.686] |
| ResNet-18, attention pool | 0.620 [0.595, 0.646] | 0.579 [0.557, 0.601] | 0.546 [0.519, 0.572] | 0.694 [0.653, 0.734] | 0.862 [0.844, 0.880] | 0.669 [0.646, 0.690] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 401 | 375 | 78 | 144 |
| ResNet-18, mean pool | 417 | 359 | 57 | 165 |
| ResNet-18, attention pool | 424 | 352 | 68 | 154 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.440 | 0.387 | 0.339 | 0.542 | 0.703 | 0.457 |
| logistic regression | Cab Franc | 0.497 | 0.752 | 0.772 | 0.222 | 0.964 | 0.857 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.532 | 0.624 | 0.720 | 0.344 | 0.761 | 0.740 |
| ResNet-18, mean pool | Baco | 0.600 | 0.532 | 0.471 | 0.729 | 0.848 | 0.605 |
| ResNet-18, mean pool | Cab Franc | 0.677 | 0.892 | 0.909 | 0.444 | 0.978 | 0.942 |
| ResNet-18, mean pool | Chardonnay | 0.556 | 0.460 | 0.212 | 0.900 | 0.791 | 0.335 |
| ResNet-18, mean pool | Vidal | 0.491 | 0.448 | 0.403 | 0.578 | 0.735 | 0.521 |
| ResNet-18, attention pool | Baco | 0.564 | 0.504 | 0.450 | 0.678 | 0.817 | 0.580 |
| ResNet-18, attention pool | Cab Franc | 0.592 | 0.832 | 0.851 | 0.333 | 0.972 | 0.907 |
| ResNet-18, attention pool | Chardonnay | 0.565 | 0.484 | 0.275 | 0.856 | 0.772 | 0.406 |
| ResNet-18, attention pool | Vidal | 0.508 | 0.496 | 0.484 | 0.531 | 0.750 | 0.588 |
