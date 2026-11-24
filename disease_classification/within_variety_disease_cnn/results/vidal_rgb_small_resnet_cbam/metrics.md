# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| small ResNet + CBAM, mean pool | 0.457 [0.406, 0.500] | 0.576 [0.548, 0.608] | 0.308 [0.231, 0.385] | 0.607 [0.576, 0.638] | 0.083 [0.059, 0.104] | 0.131 [0.094, 0.164] |
| small ResNet + CBAM, attention pool | 0.480 [0.412, 0.547] | 0.556 [0.520, 0.592] | 0.385 [0.269, 0.500] | 0.576 [0.540, 0.612] | 0.095 [0.064, 0.125] | 0.153 [0.103, 0.200] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| small ResNet + CBAM, mean pool | 8 | 18 | 88 | 136 |
| small ResNet + CBAM, attention pool | 10 | 16 | 95 | 129 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| small ResNet + CBAM, mean pool | 0.465 [0.416, 0.510] | 0.464 [0.420, 0.504] | 0.462 [0.410, 0.509] | 0.468 [0.390, 0.545] | 0.661 [0.617, 0.702] | 0.544 [0.500, 0.586] |
| small ResNet + CBAM, attention pool | 0.464 [0.418, 0.510] | 0.472 [0.432, 0.516] | 0.486 [0.439, 0.532] | 0.442 [0.364, 0.519] | 0.661 [0.623, 0.701] | 0.560 [0.519, 0.602] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| small ResNet + CBAM, mean pool | 80 | 93 | 41 | 36 |
| small ResNet + CBAM, attention pool | 84 | 89 | 43 | 34 |
