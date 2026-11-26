# Disease: logistic regression on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.422 [0.336, 0.510] | 0.452 [0.392, 0.512] | 0.385 [0.231, 0.538] | 0.460 [0.397, 0.522] | 0.076 [0.046, 0.108] | 0.127 [0.077, 0.180] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 10 | 16 | 121 | 103 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.528 [0.461, 0.589] | 0.536 [0.476, 0.596] | 0.549 [0.474, 0.618] | 0.506 [0.390, 0.610] | 0.714 [0.661, 0.764] | 0.621 [0.560, 0.675] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 95 | 78 | 38 | 39 |
