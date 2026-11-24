# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| small ResNet, mean pool | 0.513 [0.441, 0.589] | 0.524 [0.480, 0.572] | 0.500 [0.385, 0.654] | 0.527 [0.482, 0.576] | 0.109 [0.081, 0.139] | 0.179 [0.133, 0.228] |
| small ResNet, attention pool | 0.432 [0.377, 0.488] | 0.500 [0.456, 0.548] | 0.346 [0.231, 0.423] | 0.518 [0.469, 0.567] | 0.077 [0.056, 0.099] | 0.126 [0.091, 0.161] |
| small ResNet, mean ∥ max pool | 0.459 [0.405, 0.522] | 0.732 [0.688, 0.776] | 0.115 [0.000, 0.231] | 0.804 [0.759, 0.848] | 0.064 [0.000, 0.125] | 0.082 [0.000, 0.162] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| small ResNet, mean pool | 13 | 13 | 106 | 118 |
| small ResNet, attention pool | 9 | 17 | 108 | 116 |
| small ResNet, mean ∥ max pool | 3 | 23 | 44 | 180 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| small ResNet, mean pool | 0.503 [0.450, 0.556] | 0.492 [0.444, 0.540] | 0.474 [0.416, 0.532] | 0.532 [0.442, 0.623] | 0.695 [0.647, 0.743] | 0.564 [0.512, 0.613] |
| small ResNet, attention pool | 0.509 [0.456, 0.560] | 0.500 [0.452, 0.548] | 0.486 [0.428, 0.543] | 0.532 [0.455, 0.610] | 0.700 [0.654, 0.746] | 0.573 [0.518, 0.623] |
| small ResNet, mean ∥ max pool | 0.523 [0.466, 0.576] | 0.540 [0.488, 0.592] | 0.566 [0.503, 0.630] | 0.481 [0.390, 0.571] | 0.710 [0.667, 0.752] | 0.630 [0.578, 0.677] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| small ResNet, mean pool | 82 | 91 | 36 | 41 |
| small ResNet, attention pool | 84 | 89 | 36 | 41 |
| small ResNet, mean ∥ max pool | 98 | 75 | 40 | 37 |
