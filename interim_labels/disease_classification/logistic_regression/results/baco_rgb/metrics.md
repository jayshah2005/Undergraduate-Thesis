# Disease: logistic regression on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |

## Both diseases (53 + / 195 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.523 [0.455, 0.590] | 0.585 [0.528, 0.641] | 0.415 [0.302, 0.528] | 0.631 [0.564, 0.697] | 0.234 [0.174, 0.295] | 0.299 [0.221, 0.375] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 22 | 31 | 72 | 123 |

## Any disease vs healthy (189 + / 59 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.606 [0.543, 0.670] | 0.621 [0.565, 0.677] | 0.635 [0.571, 0.704] | 0.576 [0.475, 0.695] | 0.828 [0.789, 0.870] | 0.719 [0.669, 0.766] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 120 | 69 | 25 | 34 |
