# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | 0.660 [0.616, 0.700] | 0.400 [0.320, 0.480] | 0.689 [0.640, 0.733] | 0.125 [0.096, 0.157] | 0.190 [0.147, 0.235] |
| CNN, mean pool | 0.387 [0.333, 0.442] | 0.472 [0.440, 0.504] | 0.280 [0.200, 0.400] | 0.493 [0.458, 0.529] | 0.058 [0.039, 0.078] | 0.096 [0.065, 0.130] |
| CNN, attention pool | 0.478 [0.460, 0.496] | 0.508 [0.476, 0.540] | 0.440 [0.440, 0.440] | 0.516 [0.480, 0.551] | 0.092 [0.086, 0.098] | 0.152 [0.144, 0.161] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 10 | 15 | 70 | 155 |
| CNN, mean pool | 7 | 18 | 114 | 111 |
| CNN, attention pool | 11 | 14 | 109 | 116 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | 0.612 [0.560, 0.664] | 0.722 [0.651, 0.787] | 0.383 [0.284, 0.481] | 0.709 [0.672, 0.746] | 0.716 [0.669, 0.758] |
| CNN, mean pool | 0.531 [0.479, 0.579] | 0.500 [0.452, 0.548] | 0.444 [0.385, 0.509] | 0.617 [0.531, 0.691] | 0.708 [0.654, 0.757] | 0.545 [0.487, 0.603] |
| CNN, attention pool | 0.493 [0.447, 0.537] | 0.440 [0.396, 0.484] | 0.343 [0.290, 0.396] | 0.642 [0.568, 0.716] | 0.667 [0.609, 0.722] | 0.453 [0.395, 0.508] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 47 | 50 | 31 |
| CNN, mean pool | 75 | 94 | 31 | 50 |
| CNN, attention pool | 58 | 111 | 29 | 52 |
