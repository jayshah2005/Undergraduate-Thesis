# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | 0.660 [0.616, 0.700] | 0.400 [0.320, 0.480] | 0.689 [0.640, 0.733] | 0.125 [0.096, 0.157] | 0.190 [0.147, 0.235] |
| ResNet-18, mean pool | 0.396 [0.313, 0.484] | 0.456 [0.404, 0.508] | 0.320 [0.160, 0.480] | 0.471 [0.418, 0.524] | 0.063 [0.033, 0.094] | 0.105 [0.054, 0.158] |
| ResNet-18, attention pool | 0.427 [0.356, 0.504] | 0.512 [0.468, 0.560] | 0.320 [0.200, 0.480] | 0.533 [0.484, 0.582] | 0.071 [0.043, 0.102] | 0.116 [0.071, 0.167] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 10 | 15 | 70 | 155 |
| ResNet-18, mean pool | 8 | 17 | 119 | 106 |
| ResNet-18, attention pool | 8 | 17 | 105 | 120 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | 0.612 [0.560, 0.664] | 0.722 [0.651, 0.787] | 0.383 [0.284, 0.481] | 0.709 [0.672, 0.746] | 0.716 [0.669, 0.758] |
| ResNet-18, mean pool | 0.482 [0.430, 0.534] | 0.500 [0.448, 0.552] | 0.533 [0.467, 0.598] | 0.432 [0.346, 0.519] | 0.662 [0.618, 0.703] | 0.590 [0.537, 0.641] |
| ResNet-18, attention pool | 0.501 [0.449, 0.556] | 0.512 [0.464, 0.564] | 0.533 [0.473, 0.592] | 0.469 [0.383, 0.556] | 0.677 [0.634, 0.722] | 0.596 [0.547, 0.647] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 47 | 50 | 31 |
| ResNet-18, mean pool | 90 | 79 | 46 | 35 |
| ResNet-18, attention pool | 90 | 79 | 43 | 38 |
