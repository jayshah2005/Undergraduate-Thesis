# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | 0.460 [0.404, 0.516] | 0.360 [0.200, 0.520] | 0.471 [0.409, 0.529] | 0.070 [0.042, 0.098] | 0.118 [0.069, 0.164] |
| ResNet-18, mean pool | 0.360 [0.313, 0.413] | 0.456 [0.420, 0.496] | 0.240 [0.160, 0.360] | 0.480 [0.440, 0.524] | 0.049 [0.032, 0.068] | 0.081 [0.054, 0.113] |
| ResNet-18, attention pool | 0.384 [0.367, 0.402] | 0.404 [0.372, 0.436] | 0.360 [0.360, 0.360] | 0.409 [0.373, 0.444] | 0.063 [0.060, 0.067] | 0.108 [0.103, 0.113] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 9 | 16 | 119 | 106 |
| ResNet-18, mean pool | 6 | 19 | 117 | 108 |
| ResNet-18, attention pool | 9 | 16 | 133 | 92 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | 0.536 [0.480, 0.592] | 0.533 [0.462, 0.604] | 0.543 [0.444, 0.642] | 0.709 [0.654, 0.760] | 0.608 [0.549, 0.667] |
| ResNet-18, mean pool | 0.436 [0.386, 0.485] | 0.424 [0.380, 0.468] | 0.402 [0.349, 0.456] | 0.469 [0.383, 0.543] | 0.613 [0.562, 0.661] | 0.486 [0.433, 0.534] |
| ResNet-18, attention pool | 0.471 [0.419, 0.525] | 0.476 [0.428, 0.528] | 0.485 [0.426, 0.544] | 0.457 [0.370, 0.543] | 0.651 [0.603, 0.697] | 0.556 [0.504, 0.605] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 90 | 79 | 37 | 44 |
| ResNet-18, mean pool | 68 | 101 | 43 | 38 |
| ResNet-18, attention pool | 82 | 87 | 44 | 37 |
