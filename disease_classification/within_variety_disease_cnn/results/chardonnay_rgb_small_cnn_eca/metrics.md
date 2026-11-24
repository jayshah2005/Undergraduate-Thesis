# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| CNN + ECA, mean pool | 0.546 [0.498, 0.598] | 0.548 [0.500, 0.600] | 0.484 [0.410, 0.557] | 0.609 [0.531, 0.688] | 0.541 [0.486, 0.604] | 0.511 [0.450, 0.570] |
| CNN + ECA, attention pool | 0.563 [0.515, 0.611] | 0.564 [0.516, 0.612] | 0.533 [0.467, 0.598] | 0.594 [0.523, 0.664] | 0.556 [0.504, 0.609] | 0.544 [0.487, 0.596] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| CNN + ECA, mean pool | 59 | 63 | 50 | 78 |
| CNN + ECA, attention pool | 65 | 57 | 52 | 76 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| CNN + ECA, mean pool | 0.600 [0.552, 0.649] | 0.616 [0.572, 0.664] | 0.545 [0.466, 0.625] | 0.654 [0.599, 0.710] | 0.462 [0.408, 0.518] | 0.500 [0.440, 0.560] |
| CNN + ECA, attention pool | 0.600 [0.552, 0.649] | 0.616 [0.572, 0.664] | 0.545 [0.466, 0.625] | 0.654 [0.599, 0.710] | 0.462 [0.408, 0.519] | 0.500 [0.440, 0.558] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| CNN + ECA, mean pool | 48 | 40 | 56 | 106 |
| CNN + ECA, attention pool | 48 | 40 | 56 | 106 |
