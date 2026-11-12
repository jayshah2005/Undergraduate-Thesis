# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| small ResNet, mean pool | 0.548 [0.517, 0.580] | 0.548 [0.516, 0.580] | 0.557 [0.525, 0.590] | 0.539 [0.492, 0.594] | 0.535 [0.504, 0.569] | 0.546 [0.518, 0.575] |
| small ResNet, attention pool | 0.526 [0.502, 0.550] | 0.524 [0.500, 0.548] | 0.598 [0.566, 0.631] | 0.453 [0.414, 0.492] | 0.510 [0.490, 0.532] | 0.551 [0.527, 0.575] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| small ResNet, mean pool | 68 | 54 | 59 | 69 |
| small ResNet, attention pool | 73 | 49 | 70 | 58 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| small ResNet, mean pool | 0.516 [0.465, 0.564] | 0.528 [0.480, 0.572] | 0.477 [0.398, 0.557] | 0.556 [0.494, 0.611] | 0.368 [0.317, 0.417] | 0.416 [0.356, 0.472] |
| small ResNet, attention pool | 0.545 [0.498, 0.591] | 0.568 [0.528, 0.608] | 0.466 [0.386, 0.545] | 0.623 [0.574, 0.667] | 0.402 [0.350, 0.452] | 0.432 [0.370, 0.490] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| small ResNet, mean pool | 42 | 46 | 72 | 90 |
| small ResNet, attention pool | 41 | 47 | 61 | 101 |
