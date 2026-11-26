# Disease: logistic regression on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.491 [0.431, 0.551] | 0.492 [0.432, 0.552] | 0.434 [0.352, 0.516] | 0.547 [0.461, 0.633] | 0.477 [0.409, 0.546] | 0.455 [0.381, 0.524] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 69 | 58 | 70 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.634 [0.575, 0.694] | 0.640 [0.580, 0.700] | 0.614 [0.523, 0.705] | 0.654 [0.580, 0.728] | 0.491 [0.427, 0.562] | 0.545 [0.476, 0.614] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 54 | 34 | 56 | 106 |
