# Disease: logistic regression on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | 0.628 [0.606, 0.651] | 0.594 [0.564, 0.624] | 0.688 [0.655, 0.721] | 0.767 [0.746, 0.789] | 0.670 [0.646, 0.693] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 376 | 257 | 114 | 251 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.458 | 0.448 | 0.341 | 0.575 | 0.489 | 0.402 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.567 | 0.688 | 0.159 | 0.975 | 0.778 | 0.264 |
| logistic regression | Vidal | 0.542 | 0.620 | 0.763 | 0.321 | 0.701 | 0.731 |

## Both diseases (115 + / 633 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.492 [0.456, 0.529] | 0.471 [0.444, 0.497] | 0.522 [0.452, 0.591] | 0.461 [0.431, 0.490] | 0.150 [0.132, 0.168] | 0.233 [0.205, 0.260] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 60 | 55 | 341 | 292 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.484 | 0.415 | 0.503 | 0.185 | 0.256 |
| logistic regression | Chardonnay | 0.405 | 0.252 | 0.660 | 0.150 | 0.163 | 0.261 |
| logistic regression | Vidal | 0.553 | 0.676 | 0.417 | 0.689 | 0.063 | 0.110 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | 0.545 [0.521, 0.570] | 0.513 [0.486, 0.540] | 0.655 [0.611, 0.699] | 0.835 [0.816, 0.856] | 0.636 [0.611, 0.660] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 376 | 78 | 148 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.449 | 0.391 | 0.339 | 0.559 | 0.711 | 0.459 |
| logistic regression | Cab Franc | 0.495 | 0.748 | 0.768 | 0.222 | 0.964 | 0.855 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.541 | 0.620 | 0.714 | 0.368 | 0.751 | 0.732 |
