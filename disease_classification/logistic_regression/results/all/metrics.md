# Disease: logistic regression on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.637 [0.616, 0.659] | 0.634 [0.614, 0.655] | 0.626 [0.599, 0.652] | 0.649 [0.614, 0.682] | 0.756 [0.737, 0.775] | 0.685 [0.664, 0.704] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 237 | 128 | 237 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.437 | 0.419 | 0.237 | 0.637 | 0.438 | 0.308 |
| logistic regression | Cab Franc | 0.473 | 0.808 | 0.834 | 0.111 | 0.962 | 0.893 |
| logistic regression | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| logistic regression | Vidal | 0.515 | 0.644 | 0.882 | 0.148 | 0.683 | 0.770 |

## Both diseases (115 + / 633 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.533 [0.492, 0.576] | 0.667 [0.639, 0.698] | 0.339 [0.261, 0.417] | 0.727 [0.697, 0.758] | 0.184 [0.146, 0.224] | 0.239 [0.189, 0.290] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 39 | 76 | 173 | 460 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.445 | 0.484 | 0.377 | 0.513 | 0.174 | 0.238 |
| logistic regression | Chardonnay | 0.535 | 0.628 | 0.380 | 0.690 | 0.235 | 0.290 |
| logistic regression | Vidal | 0.466 | 0.888 | 0.000 | 0.933 | 0.000 | 0.000 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.587] | 0.581 [0.555, 0.607] | 0.602 [0.573, 0.631] | 0.509 [0.451, 0.562] | 0.807 [0.788, 0.827] | 0.690 [0.667, 0.712] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 465 | 307 | 111 | 115 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.431 | 0.480 | 0.524 | 0.339 | 0.717 | 0.606 |
| logistic regression | Cab Franc | 0.435 | 0.736 | 0.759 | 0.111 | 0.958 | 0.847 |
| logistic regression | Chardonnay | 0.491 | 0.436 | 0.294 | 0.689 | 0.627 | 0.400 |
| logistic regression | Vidal | 0.609 | 0.672 | 0.747 | 0.471 | 0.791 | 0.768 |
