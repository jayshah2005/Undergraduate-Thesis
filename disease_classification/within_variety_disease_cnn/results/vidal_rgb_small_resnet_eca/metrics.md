# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| small ResNet + ECA, mean pool | 0.464 [0.409, 0.529] | 0.436 [0.396, 0.476] | 0.500 [0.385, 0.615] | 0.429 [0.388, 0.469] | 0.092 [0.074, 0.113] | 0.156 [0.124, 0.192] |
| small ResNet + ECA, attention pool | 0.389 [0.321, 0.460] | 0.392 [0.348, 0.436] | 0.385 [0.269, 0.500] | 0.393 [0.348, 0.438] | 0.068 [0.047, 0.092] | 0.116 [0.080, 0.156] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| small ResNet + ECA, mean pool | 13 | 13 | 128 | 96 |
| small ResNet + ECA, attention pool | 10 | 16 | 136 | 88 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| small ResNet + ECA, mean pool | 0.461 [0.420, 0.497] | 0.448 [0.416, 0.480] | 0.428 [0.387, 0.468] | 0.494 [0.429, 0.558] | 0.655 [0.617, 0.690] | 0.517 [0.482, 0.552] |
| small ResNet + ECA, attention pool | 0.458 [0.414, 0.500] | 0.444 [0.404, 0.480] | 0.422 [0.382, 0.462] | 0.494 [0.416, 0.571] | 0.652 [0.611, 0.692] | 0.512 [0.473, 0.551] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| small ResNet + ECA, mean pool | 74 | 99 | 39 | 38 |
| small ResNet + ECA, attention pool | 73 | 100 | 39 | 38 |
