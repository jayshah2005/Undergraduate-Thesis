# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| small ResNet (no max pool), mean pool | 0.575 [0.535, 0.615] | 0.576 [0.536, 0.616] | 0.533 [0.475, 0.590] | 0.617 [0.562, 0.672] | 0.570 [0.526, 0.616] | 0.551 [0.502, 0.594] |
| small ResNet (no max pool), attention pool | 0.572 [0.536, 0.608] | 0.576 [0.540, 0.612] | 0.418 [0.361, 0.475] | 0.727 [0.680, 0.773] | 0.593 [0.540, 0.644] | 0.490 [0.438, 0.538] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| small ResNet (no max pool), mean pool | 65 | 57 | 49 | 79 |
| small ResNet (no max pool), attention pool | 51 | 71 | 35 | 93 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| small ResNet (no max pool), mean pool | 0.584 [0.526, 0.641] | 0.632 [0.580, 0.684] | 0.420 [0.318, 0.523] | 0.747 [0.685, 0.809] | 0.474 [0.391, 0.562] | 0.446 [0.357, 0.530] |
| small ResNet (no max pool), attention pool | 0.572 [0.515, 0.627] | 0.620 [0.568, 0.672] | 0.409 [0.318, 0.500] | 0.735 [0.673, 0.796] | 0.456 [0.375, 0.543] | 0.431 [0.346, 0.510] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| small ResNet (no max pool), mean pool | 37 | 51 | 41 | 121 |
| small ResNet (no max pool), attention pool | 36 | 52 | 43 | 119 |
