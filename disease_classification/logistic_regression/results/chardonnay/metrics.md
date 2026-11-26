# Disease: logistic regression on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.533 [0.473, 0.593] | 0.536 [0.476, 0.596] | 0.410 [0.328, 0.492] | 0.656 [0.570, 0.734] | 0.532 [0.453, 0.612] | 0.463 [0.383, 0.538] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 44 | 84 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.537, 0.656] | 0.612 [0.556, 0.668] | 0.545 [0.443, 0.636] | 0.648 [0.580, 0.716] | 0.457 [0.392, 0.525] | 0.497 [0.421, 0.567] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 48 | 40 | 57 | 105 |
