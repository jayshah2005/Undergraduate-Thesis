# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| ResNet-18, mean pool | 0.577 [0.551, 0.600] | 0.551 [0.525, 0.575] | 0.657 [0.618, 0.697] | 0.496 [0.464, 0.526] | 0.401 [0.381, 0.421] | 0.499 [0.473, 0.522] |
| ResNet-18, attention pool | 0.596 [0.568, 0.623] | 0.575 [0.548, 0.600] | 0.661 [0.618, 0.705] | 0.530 [0.496, 0.563] | 0.420 [0.397, 0.443] | 0.514 [0.485, 0.542] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| ResNet-18, mean pool | 167 | 87 | 249 | 245 |
| ResNet-18, attention pool | 168 | 86 | 232 | 262 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |
| ResNet-18, mean pool | Baco | 0.498 | 0.476 | 0.664 | 0.333 | 0.430 | 0.522 |
| ResNet-18, mean pool | Chardonnay | 0.463 | 0.456 | 0.746 | 0.180 | 0.464 | 0.572 |
| ResNet-18, mean pool | Vidal | 0.489 | 0.720 | 0.200 | 0.778 | 0.091 | 0.125 |
| ResNet-18, attention pool | Baco | 0.530 | 0.512 | 0.664 | 0.397 | 0.455 | 0.540 |
| ResNet-18, attention pool | Chardonnay | 0.463 | 0.456 | 0.762 | 0.164 | 0.465 | 0.578 |
| ResNet-18, attention pool | Vidal | 0.491 | 0.756 | 0.160 | 0.822 | 0.091 | 0.116 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | 0.628 [0.606, 0.651] | 0.594 [0.564, 0.624] | 0.688 [0.655, 0.721] | 0.767 [0.746, 0.789] | 0.670 [0.646, 0.693] |
| ResNet-18, mean pool | 0.659 [0.639, 0.678] | 0.631 [0.611, 0.651] | 0.556 [0.531, 0.581] | 0.762 [0.732, 0.792] | 0.802 [0.781, 0.823] | 0.657 [0.635, 0.678] |
| ResNet-18, attention pool | 0.663 [0.644, 0.683] | 0.624 [0.606, 0.644] | 0.518 [0.493, 0.545] | 0.808 [0.778, 0.838] | 0.824 [0.801, 0.848] | 0.636 [0.615, 0.659] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 376 | 257 | 114 | 251 |
| ResNet-18, mean pool | 352 | 281 | 87 | 278 |
| ResNet-18, attention pool | 328 | 305 | 70 | 295 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.458 | 0.448 | 0.341 | 0.575 | 0.489 | 0.402 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.567 | 0.688 | 0.159 | 0.975 | 0.778 | 0.264 |
| logistic regression | Vidal | 0.542 | 0.620 | 0.763 | 0.321 | 0.701 | 0.731 |
| ResNet-18, mean pool | Baco | 0.521 | 0.496 | 0.237 | 0.805 | 0.593 | 0.339 |
| ResNet-18, mean pool | Cab Franc | 0.504 | 0.868 | 0.896 | 0.111 | 0.964 | 0.929 |
| ResNet-18, mean pool | Chardonnay | 0.559 | 0.664 | 0.205 | 0.914 | 0.562 | 0.300 |
| ResNet-18, mean pool | Vidal | 0.489 | 0.496 | 0.509 | 0.469 | 0.667 | 0.577 |
| ResNet-18, attention pool | Baco | 0.491 | 0.460 | 0.133 | 0.850 | 0.514 | 0.212 |
| ResNet-18, attention pool | Cab Franc | 0.551 | 0.856 | 0.880 | 0.222 | 0.968 | 0.922 |
| ResNet-18, attention pool | Chardonnay | 0.535 | 0.656 | 0.125 | 0.944 | 0.550 | 0.204 |
| ResNet-18, attention pool | Vidal | 0.529 | 0.524 | 0.515 | 0.543 | 0.702 | 0.594 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | 0.545 [0.521, 0.570] | 0.513 [0.486, 0.540] | 0.655 [0.611, 0.699] | 0.835 [0.816, 0.856] | 0.636 [0.611, 0.660] |
| ResNet-18, mean pool | 0.621 [0.597, 0.644] | 0.573 [0.552, 0.594] | 0.534 [0.509, 0.558] | 0.708 [0.668, 0.748] | 0.862 [0.844, 0.879] | 0.659 [0.637, 0.680] |
| ResNet-18, attention pool | 0.604 [0.580, 0.627] | 0.595 [0.575, 0.615] | 0.588 [0.565, 0.611] | 0.619 [0.580, 0.659] | 0.841 [0.826, 0.856] | 0.692 [0.673, 0.711] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 376 | 78 | 148 |
| ResNet-18, mean pool | 412 | 360 | 66 | 160 |
| ResNet-18, attention pool | 454 | 318 | 86 | 140 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.449 | 0.391 | 0.339 | 0.559 | 0.711 | 0.459 |
| logistic regression | Cab Franc | 0.495 | 0.748 | 0.768 | 0.222 | 0.964 | 0.855 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.541 | 0.620 | 0.714 | 0.368 | 0.751 | 0.732 |
| ResNet-18, mean pool | Baco | 0.518 | 0.399 | 0.291 | 0.746 | 0.786 | 0.425 |
| ResNet-18, mean pool | Cab Franc | 0.555 | 0.864 | 0.888 | 0.222 | 0.968 | 0.926 |
| ResNet-18, mean pool | Chardonnay | 0.579 | 0.520 | 0.369 | 0.789 | 0.756 | 0.496 |
| ResNet-18, mean pool | Vidal | 0.547 | 0.508 | 0.462 | 0.632 | 0.771 | 0.577 |
| ResNet-18, attention pool | Baco | 0.548 | 0.516 | 0.487 | 0.610 | 0.800 | 0.605 |
| ResNet-18, attention pool | Cab Franc | 0.598 | 0.844 | 0.863 | 0.333 | 0.972 | 0.914 |
| ResNet-18, attention pool | Chardonnay | 0.542 | 0.488 | 0.350 | 0.733 | 0.700 | 0.467 |
| ResNet-18, attention pool | Vidal | 0.527 | 0.532 | 0.538 | 0.515 | 0.748 | 0.626 |
