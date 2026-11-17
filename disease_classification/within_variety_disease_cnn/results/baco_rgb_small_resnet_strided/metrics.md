# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| small ResNet (no max pool), mean pool | 0.497 [0.459, 0.532] | 0.520 [0.484, 0.556] | 0.327 [0.271, 0.383] | 0.667 [0.624, 0.709] | 0.427 [0.370, 0.481] | 0.370 [0.314, 0.423] |
| small ResNet (no max pool), attention pool | 0.534 [0.505, 0.561] | 0.536 [0.508, 0.565] | 0.514 [0.477, 0.551] | 0.553 [0.504, 0.596] | 0.466 [0.437, 0.496] | 0.489 [0.462, 0.520] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| small ResNet (no max pool), mean pool | 35 | 72 | 47 | 94 |
| small ResNet (no max pool), attention pool | 55 | 52 | 63 | 78 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| small ResNet (no max pool), mean pool | 0.563 [0.521, 0.607] | 0.569 [0.528, 0.613] | 0.622 [0.570, 0.681] | 0.504 [0.442, 0.566] | 0.600 [0.563, 0.639] | 0.611 [0.569, 0.655] |
| small ResNet (no max pool), attention pool | 0.534 [0.492, 0.576] | 0.524 [0.484, 0.569] | 0.422 [0.363, 0.481] | 0.646 [0.584, 0.708] | 0.588 [0.535, 0.640] | 0.491 [0.435, 0.545] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| small ResNet (no max pool), mean pool | 84 | 51 | 56 | 57 |
| small ResNet (no max pool), attention pool | 57 | 78 | 40 | 73 |
