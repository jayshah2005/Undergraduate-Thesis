# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| ResNet-18, mean pool | 0.395 [0.339, 0.447] | 0.464 [0.420, 0.504] | 0.308 [0.230, 0.385] | 0.482 [0.438, 0.527] | 0.065 [0.044, 0.084] | 0.107 [0.073, 0.139] |
| ResNet-18, attention pool | 0.378 [0.331, 0.430] | 0.464 [0.424, 0.504] | 0.269 [0.192, 0.346] | 0.487 [0.446, 0.527] | 0.057 [0.040, 0.077] | 0.095 [0.067, 0.126] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| ResNet-18, mean pool | 8 | 18 | 116 | 108 |
| ResNet-18, attention pool | 7 | 19 | 115 | 109 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| ResNet-18, mean pool | 0.509 [0.451, 0.564] | 0.560 [0.508, 0.612] | 0.642 [0.578, 0.699] | 0.377 [0.286, 0.468] | 0.698 [0.660, 0.735] | 0.669 [0.621, 0.712] |
| ResNet-18, attention pool | 0.511 [0.474, 0.549] | 0.488 [0.452, 0.524] | 0.451 [0.405, 0.497] | 0.571 [0.506, 0.623] | 0.703 [0.667, 0.739] | 0.549 [0.507, 0.590] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| ResNet-18, mean pool | 111 | 62 | 48 | 29 |
| ResNet-18, attention pool | 78 | 95 | 33 | 44 |
