# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| small ResNet + ECA, mean pool | 0.490 [0.444, 0.535] | 0.500 [0.452, 0.544] | 0.421 [0.355, 0.486] | 0.560 [0.496, 0.617] | 0.421 [0.369, 0.472] | 0.421 [0.364, 0.476] |
| small ResNet + ECA, attention pool | 0.454 [0.409, 0.500] | 0.460 [0.415, 0.504] | 0.411 [0.336, 0.477] | 0.496 [0.440, 0.553] | 0.383 [0.333, 0.431] | 0.396 [0.338, 0.451] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| small ResNet + ECA, mean pool | 45 | 62 | 62 | 79 |
| small ResNet + ECA, attention pool | 44 | 63 | 71 | 70 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| small ResNet + ECA, mean pool | 0.570 [0.527, 0.611] | 0.548 [0.504, 0.589] | 0.326 [0.267, 0.378] | 0.814 [0.752, 0.876] | 0.677 [0.594, 0.761] | 0.440 [0.374, 0.498] |
| small ResNet + ECA, attention pool | 0.554 [0.508, 0.600] | 0.548 [0.504, 0.593] | 0.489 [0.422, 0.556] | 0.619 [0.558, 0.681] | 0.606 [0.554, 0.655] | 0.541 [0.487, 0.594] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| small ResNet + ECA, mean pool | 44 | 91 | 21 | 92 |
| small ResNet + ECA, attention pool | 66 | 69 | 43 | 70 |
