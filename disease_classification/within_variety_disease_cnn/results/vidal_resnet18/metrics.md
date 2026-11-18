# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| ResNet-18, mean pool | 0.440 [0.365, 0.518] | 0.544 [0.492, 0.592] | 0.308 [0.192, 0.462] | 0.571 [0.518, 0.621] | 0.077 [0.044, 0.111] | 0.123 [0.072, 0.179] |
| ResNet-18, attention pool | 0.440 [0.368, 0.517] | 0.484 [0.440, 0.528] | 0.385 [0.269, 0.538] | 0.496 [0.446, 0.540] | 0.081 [0.055, 0.110] | 0.134 [0.091, 0.183] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| ResNet-18, mean pool | 8 | 18 | 96 | 128 |
| ResNet-18, attention pool | 10 | 16 | 113 | 111 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| ResNet-18, mean pool | 0.533 [0.479, 0.586] | 0.508 [0.456, 0.560] | 0.468 [0.410, 0.532] | 0.597 [0.506, 0.688] | 0.723 [0.672, 0.775] | 0.568 [0.513, 0.623] |
| ResNet-18, attention pool | 0.510 [0.462, 0.559] | 0.516 [0.472, 0.560] | 0.526 [0.474, 0.578] | 0.494 [0.416, 0.571] | 0.700 [0.661, 0.740] | 0.601 [0.556, 0.645] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| ResNet-18, mean pool | 81 | 92 | 31 | 46 |
| ResNet-18, attention pool | 91 | 82 | 39 | 38 |
