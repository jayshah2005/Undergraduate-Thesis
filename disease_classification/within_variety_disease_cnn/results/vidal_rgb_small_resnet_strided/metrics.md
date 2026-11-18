# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| small ResNet (no max pool), mean pool | 0.431 [0.359, 0.507] | 0.316 [0.276, 0.356] | 0.577 [0.423, 0.731] | 0.286 [0.246, 0.326] | 0.086 [0.066, 0.106] | 0.149 [0.114, 0.184] |
| small ResNet (no max pool), attention pool | 0.411 [0.400, 0.424] | 0.432 [0.412, 0.456] | 0.385 [0.385, 0.385] | 0.438 [0.415, 0.464] | 0.074 [0.071, 0.077] | 0.123 [0.120, 0.128] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| small ResNet (no max pool), mean pool | 15 | 11 | 160 | 64 |
| small ResNet (no max pool), attention pool | 10 | 16 | 126 | 98 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| small ResNet (no max pool), mean pool | 0.479 [0.428, 0.528] | 0.468 [0.424, 0.512] | 0.451 [0.393, 0.503] | 0.506 [0.429, 0.584] | 0.672 [0.626, 0.718] | 0.540 [0.489, 0.586] |
| small ResNet (no max pool), attention pool | 0.498 [0.446, 0.545] | 0.520 [0.476, 0.564] | 0.555 [0.503, 0.607] | 0.442 [0.364, 0.519] | 0.691 [0.651, 0.726] | 0.615 [0.570, 0.658] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| small ResNet (no max pool), mean pool | 78 | 95 | 38 | 39 |
| small ResNet (no max pool), attention pool | 96 | 77 | 43 | 34 |
