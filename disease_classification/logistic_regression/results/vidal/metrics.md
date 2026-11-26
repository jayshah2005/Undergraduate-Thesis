# Disease: logistic regression on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.457 [0.370, 0.543] | 0.484 [0.424, 0.540] | 0.423 [0.269, 0.577] | 0.491 [0.433, 0.554] | 0.088 [0.056, 0.120] | 0.146 [0.093, 0.199] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 114 | 110 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.476, 0.601] | 0.560 [0.504, 0.620] | 0.595 [0.526, 0.665] | 0.481 [0.377, 0.584] | 0.720 [0.674, 0.769] | 0.652 [0.597, 0.705] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 103 | 70 | 40 | 37 |
