# Disease: logistic regression on Baco + Chardonnay + Vidal, RGB bands only: all metrics

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.610 [0.588, 0.633] | 0.614 [0.591, 0.636] | 0.667 [0.634, 0.697] | 0.554 [0.523, 0.585] | 0.627 [0.608, 0.647] | 0.646 [0.624, 0.669] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 264 | 132 | 157 | 195 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.472 | 0.593 | 0.327 | 0.513 | 0.550 |
| logistic regression | Chardonnay | 0.563 | 0.676 | 0.182 | 0.944 | 0.640 | 0.283 |
| logistic regression | Vidal | 0.518 | 0.692 | 0.971 | 0.065 | 0.700 | 0.814 |
