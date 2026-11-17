# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | 0.460 [0.404, 0.516] | 0.360 [0.200, 0.520] | 0.471 [0.409, 0.529] | 0.070 [0.042, 0.098] | 0.118 [0.069, 0.164] |
| small ResNet (no max pool), mean pool | 0.353 [0.284, 0.433] | 0.380 [0.336, 0.424] | 0.320 [0.200, 0.480] | 0.387 [0.338, 0.431] | 0.055 [0.034, 0.079] | 0.094 [0.058, 0.136] |
| small ResNet (no max pool), attention pool | 0.458 [0.442, 0.473] | 0.472 [0.444, 0.500] | 0.440 [0.440, 0.440] | 0.476 [0.444, 0.507] | 0.085 [0.081, 0.090] | 0.143 [0.137, 0.150] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 9 | 16 | 119 | 106 |
| small ResNet (no max pool), mean pool | 8 | 17 | 138 | 87 |
| small ResNet (no max pool), attention pool | 11 | 14 | 118 | 107 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | 0.536 [0.480, 0.592] | 0.533 [0.462, 0.604] | 0.543 [0.444, 0.642] | 0.709 [0.654, 0.760] | 0.608 [0.549, 0.667] |
| small ResNet (no max pool), mean pool | 0.503 [0.449, 0.557] | 0.476 [0.424, 0.528] | 0.426 [0.367, 0.491] | 0.580 [0.494, 0.667] | 0.679 [0.624, 0.737] | 0.524 [0.464, 0.582] |
| small ResNet (no max pool), attention pool | 0.480 [0.438, 0.523] | 0.480 [0.440, 0.520] | 0.479 [0.426, 0.533] | 0.481 [0.420, 0.543] | 0.659 [0.620, 0.696] | 0.555 [0.509, 0.599] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 90 | 79 | 37 | 44 |
| small ResNet (no max pool), mean pool | 72 | 97 | 34 | 47 |
| small ResNet (no max pool), attention pool | 81 | 88 | 42 | 39 |
