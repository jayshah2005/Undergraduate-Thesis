# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| small ResNet + CBAM, mean pool | 0.567 [0.520, 0.615] | 0.568 [0.520, 0.616] | 0.541 [0.475, 0.615] | 0.594 [0.523, 0.656] | 0.559 [0.509, 0.611] | 0.550 [0.496, 0.606] |
| small ResNet + CBAM, attention pool | 0.540 [0.492, 0.587] | 0.544 [0.496, 0.592] | 0.361 [0.287, 0.434] | 0.719 [0.648, 0.781] | 0.550 [0.476, 0.627] | 0.436 [0.364, 0.505] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| small ResNet + CBAM, mean pool | 66 | 56 | 52 | 76 |
| small ResNet + CBAM, attention pool | 44 | 78 | 36 | 92 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| small ResNet + CBAM, mean pool | 0.569 [0.513, 0.628] | 0.620 [0.568, 0.672] | 0.398 [0.307, 0.489] | 0.741 [0.679, 0.802] | 0.455 [0.371, 0.544] | 0.424 [0.337, 0.506] |
| small ResNet + CBAM, attention pool | 0.586 [0.527, 0.645] | 0.628 [0.576, 0.684] | 0.443 [0.341, 0.545] | 0.728 [0.667, 0.796] | 0.470 [0.389, 0.558] | 0.456 [0.369, 0.539] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| small ResNet + CBAM, mean pool | 35 | 53 | 42 | 120 |
| small ResNet + CBAM, attention pool | 39 | 49 | 44 | 118 |
