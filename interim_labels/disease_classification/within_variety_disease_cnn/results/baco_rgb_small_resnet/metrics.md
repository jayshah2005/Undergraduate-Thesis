# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| small ResNet, mean pool | 0.535 [0.511, 0.559] | 0.544 [0.520, 0.569] | 0.467 [0.430, 0.505] | 0.603 [0.567, 0.631] | 0.472 [0.444, 0.500] | 0.469 [0.441, 0.500] |
| small ResNet, attention pool | 0.489 [0.453, 0.524] | 0.464 [0.427, 0.500] | 0.673 [0.617, 0.729] | 0.305 [0.262, 0.355] | 0.424 [0.398, 0.448] | 0.520 [0.487, 0.552] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| small ResNet, mean pool | 50 | 57 | 56 | 85 |
| small ResNet, attention pool | 72 | 35 | 98 | 43 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| small ResNet, mean pool | 0.542 [0.497, 0.585] | 0.536 [0.492, 0.581] | 0.474 [0.407, 0.541] | 0.611 [0.549, 0.673] | 0.593 [0.541, 0.644] | 0.527 [0.468, 0.580] |
| small ResNet, attention pool | 0.573 [0.533, 0.612] | 0.560 [0.520, 0.601] | 0.430 [0.378, 0.489] | 0.717 [0.655, 0.770] | 0.644 [0.589, 0.700] | 0.516 [0.463, 0.569] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| small ResNet, mean pool | 64 | 71 | 44 | 69 |
| small ResNet, attention pool | 58 | 77 | 32 | 81 |
