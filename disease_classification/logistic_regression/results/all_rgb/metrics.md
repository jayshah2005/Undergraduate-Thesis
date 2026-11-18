# Disease: logistic regression on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |

## Both diseases (116 + / 632 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.499 [0.464, 0.534] | 0.475 [0.448, 0.501] | 0.534 [0.466, 0.595] | 0.464 [0.434, 0.492] | 0.155 [0.137, 0.172] | 0.240 [0.212, 0.266] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 62 | 54 | 339 | 293 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.492 | 0.434 | 0.508 | 0.193 | 0.267 |
| logistic regression | Chardonnay | 0.425 | 0.272 | 0.680 | 0.170 | 0.170 | 0.272 |
| logistic regression | Vidal | 0.530 | 0.660 | 0.385 | 0.675 | 0.061 | 0.105 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | 0.546 [0.523, 0.571] | 0.517 [0.491, 0.544] | 0.649 [0.599, 0.694] | 0.837 [0.817, 0.857] | 0.639 [0.616, 0.663] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 401 | 375 | 78 | 144 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.440 | 0.387 | 0.339 | 0.542 | 0.703 | 0.457 |
| logistic regression | Cab Franc | 0.497 | 0.752 | 0.772 | 0.222 | 0.964 | 0.857 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.532 | 0.624 | 0.720 | 0.344 | 0.761 | 0.740 |
