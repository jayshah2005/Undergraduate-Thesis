# Disease: logistic regression on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.573 [0.546, 0.599] | 0.594 [0.570, 0.618] | 0.510 [0.463, 0.553] | 0.637 [0.611, 0.663] | 0.421 [0.392, 0.449] | 0.461 [0.426, 0.493] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 130 | 125 | 179 | 314 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.458 | 0.484 | 0.271 | 0.645 | 0.367 | 0.312 |
| logistic regression | Chardonnay | 0.445 | 0.436 | 0.820 | 0.070 | 0.457 | 0.587 |
| logistic regression | Vidal | 0.497 | 0.860 | 0.038 | 0.955 | 0.091 | 0.054 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | 0.616 [0.593, 0.639] | 0.551 [0.520, 0.581] | 0.731 [0.698, 0.765] | 0.783 [0.760, 0.807] | 0.647 [0.620, 0.672] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 351 | 286 | 97 | 264 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.478 | 0.464 | 0.319 | 0.637 | 0.512 | 0.393 |
| logistic regression | Cab Franc | 0.542 | 0.736 | 0.751 | 0.333 | 0.968 | 0.846 |
| logistic regression | Chardonnay | 0.542 | 0.672 | 0.102 | 0.981 | 0.750 | 0.180 |
| logistic regression | Vidal | 0.536 | 0.592 | 0.682 | 0.390 | 0.715 | 0.698 |
