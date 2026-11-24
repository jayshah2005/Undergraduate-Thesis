# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| small ResNet + ECA, mean pool | 0.576 [0.527, 0.621] | 0.580 [0.532, 0.624] | 0.393 [0.320, 0.467] | 0.758 [0.695, 0.820] | 0.608 [0.533, 0.681] | 0.478 [0.406, 0.542] |
| small ResNet + ECA, attention pool | 0.581 [0.542, 0.617] | 0.584 [0.544, 0.620] | 0.459 [0.402, 0.508] | 0.703 [0.648, 0.750] | 0.596 [0.545, 0.648] | 0.519 [0.469, 0.565] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| small ResNet + ECA, mean pool | 48 | 74 | 31 | 97 |
| small ResNet + ECA, attention pool | 56 | 66 | 38 | 90 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| small ResNet + ECA, mean pool | 0.587 [0.528, 0.646] | 0.636 [0.584, 0.692] | 0.420 [0.318, 0.523] | 0.753 [0.691, 0.815] | 0.481 [0.395, 0.571] | 0.448 [0.361, 0.533] |
| small ResNet + ECA, attention pool | 0.589 [0.532, 0.647] | 0.636 [0.584, 0.688] | 0.432 [0.341, 0.534] | 0.747 [0.685, 0.809] | 0.481 [0.400, 0.570] | 0.455 [0.371, 0.537] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| small ResNet + ECA, mean pool | 37 | 51 | 40 | 122 |
| small ResNet + ECA, attention pool | 38 | 50 | 41 | 121 |
