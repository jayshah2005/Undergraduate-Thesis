# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | 0.460 [0.404, 0.516] | 0.360 [0.200, 0.520] | 0.471 [0.409, 0.529] | 0.070 [0.042, 0.098] | 0.118 [0.069, 0.164] |
| small ResNet, mean pool | 0.376 [0.320, 0.422] | 0.388 [0.352, 0.424] | 0.360 [0.280, 0.440] | 0.391 [0.351, 0.431] | 0.062 [0.046, 0.076] | 0.105 [0.079, 0.129] |
| small ResNet, attention pool | 0.340 [0.291, 0.376] | 0.356 [0.320, 0.392] | 0.320 [0.240, 0.360] | 0.360 [0.320, 0.400] | 0.053 [0.039, 0.061] | 0.090 [0.067, 0.105] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 9 | 16 | 119 | 106 |
| small ResNet, mean pool | 9 | 16 | 137 | 88 |
| small ResNet, attention pool | 8 | 17 | 144 | 81 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | 0.536 [0.480, 0.592] | 0.533 [0.462, 0.604] | 0.543 [0.444, 0.642] | 0.709 [0.654, 0.760] | 0.608 [0.549, 0.667] |
| small ResNet, mean pool | 0.484 [0.441, 0.526] | 0.480 [0.444, 0.520] | 0.473 [0.432, 0.515] | 0.494 [0.420, 0.568] | 0.661 [0.624, 0.698] | 0.552 [0.514, 0.592] |
| small ResNet, attention pool | 0.510 [0.459, 0.564] | 0.512 [0.464, 0.560] | 0.515 [0.456, 0.574] | 0.506 [0.420, 0.593] | 0.685 [0.641, 0.731] | 0.588 [0.537, 0.636] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 90 | 79 | 37 | 44 |
| small ResNet, mean pool | 80 | 89 | 41 | 40 |
| small ResNet, attention pool | 87 | 82 | 40 | 41 |
