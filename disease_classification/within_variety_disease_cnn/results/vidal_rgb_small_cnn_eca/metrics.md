# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| CNN + ECA, mean pool | 0.458 [0.402, 0.522] | 0.424 [0.384, 0.460] | 0.500 [0.385, 0.615] | 0.415 [0.375, 0.455] | 0.090 [0.072, 0.111] | 0.153 [0.122, 0.188] |
| CNN + ECA, attention pool | 0.395 [0.319, 0.472] | 0.312 [0.268, 0.356] | 0.500 [0.346, 0.654] | 0.290 [0.245, 0.335] | 0.076 [0.054, 0.097] | 0.131 [0.094, 0.168] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| CNN + ECA, mean pool | 13 | 13 | 131 | 93 |
| CNN + ECA, attention pool | 13 | 13 | 159 | 65 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| CNN + ECA, mean pool | 0.465 [0.420, 0.508] | 0.464 [0.424, 0.504] | 0.462 [0.416, 0.509] | 0.468 [0.390, 0.545] | 0.661 [0.622, 0.699] | 0.544 [0.502, 0.585] |
| CNN + ECA, attention pool | 0.490 [0.438, 0.540] | 0.488 [0.440, 0.536] | 0.486 [0.428, 0.543] | 0.494 [0.403, 0.571] | 0.683 [0.638, 0.727] | 0.568 [0.514, 0.616] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| CNN + ECA, mean pool | 80 | 93 | 41 | 36 |
| CNN + ECA, attention pool | 84 | 89 | 39 | 38 |
