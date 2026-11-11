# Disease: logistic regression on Baco + Chardonnay + Vidal, full spectrum: all metrics

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

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.468 | 0.548 | 0.372 | 0.510 | 0.529 |
| logistic regression | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| logistic regression | Vidal | 0.546 | 0.668 | 0.893 | 0.198 | 0.699 | 0.784 |

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

## Any disease vs healthy (531 + / 217 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.523 [0.488, 0.558] | 0.543 [0.511, 0.575] | 0.571 [0.533, 0.608] | 0.475 [0.419, 0.535] | 0.727 [0.701, 0.753] | 0.639 [0.609, 0.669] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 303 | 228 | 114 | 103 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.446 | 0.565 | 0.672 | 0.220 | 0.734 | 0.702 |
| logistic regression | Chardonnay | 0.494 | 0.468 | 0.400 | 0.589 | 0.634 | 0.490 |
| logistic regression | Vidal | 0.580 | 0.596 | 0.615 | 0.544 | 0.783 | 0.689 |
