# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| small ResNet, mean pool | 0.368 [0.323, 0.400] | 0.324 [0.288, 0.360] | 0.423 [0.346, 0.462] | 0.312 [0.272, 0.353] | 0.067 [0.054, 0.075] | 0.115 [0.094, 0.129] |
| small ResNet, attention pool | 0.392 [0.347, 0.421] | 0.368 [0.340, 0.396] | 0.423 [0.346, 0.462] | 0.362 [0.330, 0.393] | 0.071 [0.058, 0.079] | 0.122 [0.099, 0.136] |
| small ResNet, mean ∥ max pool | 0.467 [0.432, 0.513] | 0.440 [0.404, 0.476] | 0.500 [0.462, 0.577] | 0.433 [0.393, 0.473] | 0.093 [0.082, 0.109] | 0.157 [0.140, 0.183] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| small ResNet, mean pool | 11 | 15 | 154 | 70 |
| small ResNet, attention pool | 11 | 15 | 143 | 81 |
| small ResNet, mean ∥ max pool | 13 | 13 | 127 | 97 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| small ResNet, mean pool | 0.482 [0.435, 0.528] | 0.468 [0.428, 0.508] | 0.445 [0.399, 0.491] | 0.519 [0.442, 0.597] | 0.675 [0.632, 0.718] | 0.537 [0.494, 0.578] |
| small ResNet, attention pool | 0.483 [0.437, 0.528] | 0.484 [0.444, 0.528] | 0.486 [0.434, 0.538] | 0.481 [0.403, 0.558] | 0.677 [0.637, 0.715] | 0.566 [0.521, 0.611] |
| small ResNet, mean ∥ max pool | 0.458 [0.420, 0.496] | 0.460 [0.428, 0.492] | 0.462 [0.428, 0.503] | 0.455 [0.390, 0.519] | 0.656 [0.622, 0.689] | 0.542 [0.510, 0.575] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| small ResNet, mean pool | 77 | 96 | 37 | 40 |
| small ResNet, attention pool | 84 | 89 | 40 | 37 |
| small ResNet, mean ∥ max pool | 80 | 93 | 42 | 35 |
