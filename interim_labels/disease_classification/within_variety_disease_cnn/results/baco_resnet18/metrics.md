# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| ResNet-18, mean pool | 0.566 [0.526, 0.608] | 0.565 [0.524, 0.605] | 0.579 [0.514, 0.645] | 0.553 [0.504, 0.603] | 0.496 [0.457, 0.537] | 0.534 [0.487, 0.582] |
| ResNet-18, attention pool | 0.533 [0.487, 0.578] | 0.532 [0.488, 0.577] | 0.542 [0.467, 0.607] | 0.525 [0.468, 0.582] | 0.464 [0.419, 0.508] | 0.500 [0.443, 0.549] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| ResNet-18, mean pool | 62 | 45 | 63 | 78 |
| ResNet-18, attention pool | 58 | 49 | 67 | 74 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| ResNet-18, mean pool | 0.505 [0.448, 0.559] | 0.500 [0.444, 0.553] | 0.452 [0.378, 0.533] | 0.558 [0.478, 0.637] | 0.550 [0.487, 0.612] | 0.496 [0.428, 0.561] |
| ResNet-18, attention pool | 0.471 [0.425, 0.522] | 0.464 [0.415, 0.516] | 0.385 [0.319, 0.452] | 0.558 [0.487, 0.628] | 0.510 [0.451, 0.571] | 0.439 [0.376, 0.500] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| ResNet-18, mean pool | 61 | 74 | 50 | 63 |
| ResNet-18, attention pool | 52 | 83 | 50 | 63 |
