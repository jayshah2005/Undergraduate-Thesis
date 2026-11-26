# Disease: logistic regression on Baco + Chardonnay + Vidal, full spectrum: all metrics

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.580 [0.557, 0.605] | 0.582 [0.559, 0.607] | 0.609 [0.573, 0.646] | 0.551 [0.523, 0.582] | 0.604 [0.583, 0.627] | 0.606 [0.580, 0.634] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 155 | 158 | 194 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.452 | 0.472 | 0.674 | 0.230 | 0.511 | 0.581 |
| logistic regression | Chardonnay | 0.521 | 0.652 | 0.080 | 0.963 | 0.538 | 0.139 |
| logistic regression | Vidal | 0.491 | 0.620 | 0.827 | 0.156 | 0.688 | 0.751 |
