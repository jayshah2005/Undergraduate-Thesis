# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| ResNet-18, mean pool | 0.538 [0.486, 0.586] | 0.536 [0.484, 0.584] | 0.615 [0.549, 0.680] | 0.461 [0.383, 0.539] | 0.521 [0.476, 0.565] | 0.564 [0.513, 0.610] |
| ResNet-18, attention pool | 0.485 [0.433, 0.538] | 0.484 [0.432, 0.536] | 0.541 [0.467, 0.615] | 0.430 [0.359, 0.508] | 0.475 [0.426, 0.523] | 0.506 [0.449, 0.560] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| ResNet-18, mean pool | 75 | 47 | 69 | 59 |
| ResNet-18, attention pool | 66 | 56 | 73 | 55 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| ResNet-18, mean pool | 0.503 [0.443, 0.563] | 0.528 [0.472, 0.584] | 0.420 [0.329, 0.523] | 0.586 [0.512, 0.660] | 0.356 [0.289, 0.421] | 0.385 [0.308, 0.459] |
| ResNet-18, attention pool | 0.545 [0.487, 0.597] | 0.568 [0.516, 0.620] | 0.466 [0.375, 0.557] | 0.623 [0.556, 0.685] | 0.402 [0.337, 0.464] | 0.432 [0.359, 0.497] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| ResNet-18, mean pool | 37 | 51 | 67 | 95 |
| ResNet-18, attention pool | 41 | 47 | 61 | 101 |
