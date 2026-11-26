# Disease: logistic regression on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.587 [0.557, 0.616] | 0.583 [0.557, 0.608] | 0.600 [0.549, 0.651] | 0.574 [0.546, 0.602] | 0.421 [0.395, 0.447] | 0.495 [0.460, 0.527] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 153 | 102 | 210 | 283 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.437 | 0.415 | 0.598 | 0.277 | 0.386 | 0.469 |
| logistic regression | Chardonnay | 0.474 | 0.468 | 0.721 | 0.227 | 0.471 | 0.570 |
| logistic regression | Vidal | 0.499 | 0.864 | 0.038 | 0.960 | 0.100 | 0.056 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.663 [0.643, 0.684] | 0.657 [0.637, 0.678] | 0.642 [0.615, 0.669] | 0.684 [0.654, 0.715] | 0.782 [0.764, 0.801] | 0.705 [0.686, 0.725] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 114 | 247 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.467 | 0.448 | 0.252 | 0.681 | 0.486 | 0.332 |
| logistic regression | Cab Franc | 0.668 | 0.876 | 0.892 | 0.444 | 0.977 | 0.933 |
| logistic regression | Chardonnay | 0.536 | 0.664 | 0.102 | 0.969 | 0.643 | 0.176 |
| logistic regression | Vidal | 0.495 | 0.640 | 0.873 | 0.117 | 0.689 | 0.770 |
