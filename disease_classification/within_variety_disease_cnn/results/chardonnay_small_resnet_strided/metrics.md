# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| small ResNet (no max pool), mean pool | 0.572 [0.537, 0.608] | 0.572 [0.536, 0.608] | 0.590 [0.549, 0.639] | 0.555 [0.500, 0.609] | 0.558 [0.523, 0.595] | 0.574 [0.539, 0.609] |
| small ResNet (no max pool), attention pool | 0.531 [0.507, 0.555] | 0.528 [0.504, 0.552] | 0.656 [0.631, 0.680] | 0.406 [0.367, 0.445] | 0.513 [0.494, 0.532] | 0.576 [0.556, 0.596] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| small ResNet (no max pool), mean pool | 72 | 50 | 57 | 71 |
| small ResNet (no max pool), attention pool | 80 | 42 | 76 | 52 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| small ResNet (no max pool), mean pool | 0.568 [0.515, 0.622] | 0.608 [0.556, 0.656] | 0.432 [0.341, 0.523] | 0.704 [0.642, 0.759] | 0.442 [0.372, 0.513] | 0.437 [0.358, 0.509] |
| small ResNet (no max pool), attention pool | 0.570 [0.517, 0.623] | 0.584 [0.532, 0.636] | 0.523 [0.443, 0.602] | 0.617 [0.556, 0.685] | 0.426 [0.369, 0.485] | 0.469 [0.406, 0.531] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| small ResNet (no max pool), mean pool | 38 | 50 | 48 | 114 |
| small ResNet (no max pool), attention pool | 46 | 42 | 62 | 100 |
