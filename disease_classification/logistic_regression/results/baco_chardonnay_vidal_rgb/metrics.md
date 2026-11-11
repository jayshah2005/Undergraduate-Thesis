# Disease: logistic regression on Baco + Chardonnay + Vidal, RGB bands only: all metrics

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

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |

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

## Any disease vs healthy (531 + / 217 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.564 [0.535, 0.592] | 0.576 [0.549, 0.603] | 0.593 [0.561, 0.625] | 0.535 [0.488, 0.581] | 0.757 [0.736, 0.778] | 0.665 [0.639, 0.689] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 315 | 216 | 101 | 116 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.427 | 0.508 | 0.582 | 0.271 | 0.719 | 0.643 |
| logistic regression | Chardonnay | 0.610 | 0.516 | 0.275 | 0.944 | 0.898 | 0.421 |
| logistic regression | Vidal | 0.553 | 0.704 | 0.885 | 0.221 | 0.752 | 0.813 |
