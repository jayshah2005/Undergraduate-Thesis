# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| ResNet-18, mean pool | 0.554 [0.505, 0.600] | 0.556 [0.508, 0.605] | 0.533 [0.458, 0.607] | 0.574 [0.511, 0.638] | 0.487 [0.436, 0.538] | 0.509 [0.448, 0.564] |
| ResNet-18, attention pool | 0.582 [0.543, 0.619] | 0.581 [0.544, 0.617] | 0.589 [0.523, 0.645] | 0.574 [0.525, 0.624] | 0.512 [0.474, 0.550] | 0.548 [0.502, 0.589] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| ResNet-18, mean pool | 57 | 50 | 60 | 81 |
| ResNet-18, attention pool | 63 | 44 | 60 | 81 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| ResNet-18, mean pool | 0.515 [0.463, 0.567] | 0.508 [0.456, 0.560] | 0.437 [0.363, 0.504] | 0.593 [0.522, 0.673] | 0.562 [0.500, 0.622] | 0.492 [0.423, 0.552] |
| ResNet-18, attention pool | 0.563 [0.512, 0.615] | 0.565 [0.512, 0.617] | 0.578 [0.511, 0.644] | 0.549 [0.469, 0.628] | 0.605 [0.556, 0.654] | 0.591 [0.536, 0.644] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| ResNet-18, mean pool | 59 | 76 | 46 | 67 |
| ResNet-18, attention pool | 78 | 57 | 51 | 62 |
