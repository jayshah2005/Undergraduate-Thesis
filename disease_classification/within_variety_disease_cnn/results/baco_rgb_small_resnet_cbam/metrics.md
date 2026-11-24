# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| small ResNet + CBAM, mean pool | 0.580 [0.537, 0.622] | 0.601 [0.556, 0.645] | 0.430 [0.364, 0.495] | 0.730 [0.674, 0.787] | 0.548 [0.484, 0.614] | 0.482 [0.420, 0.538] |
| small ResNet + CBAM, attention pool | 0.484 [0.426, 0.540] | 0.476 [0.419, 0.532] | 0.542 [0.458, 0.626] | 0.426 [0.348, 0.496] | 0.417 [0.365, 0.466] | 0.472 [0.408, 0.530] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| small ResNet + CBAM, mean pool | 46 | 61 | 38 | 103 |
| small ResNet + CBAM, attention pool | 58 | 49 | 81 | 60 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| small ResNet + CBAM, mean pool | 0.530 [0.482, 0.573] | 0.516 [0.468, 0.560] | 0.378 [0.311, 0.444] | 0.681 [0.611, 0.743] | 0.586 [0.519, 0.650] | 0.459 [0.393, 0.519] |
| small ResNet + CBAM, attention pool | 0.554 [0.506, 0.601] | 0.548 [0.500, 0.597] | 0.489 [0.422, 0.556] | 0.619 [0.549, 0.690] | 0.606 [0.551, 0.658] | 0.541 [0.483, 0.596] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| small ResNet + CBAM, mean pool | 51 | 84 | 36 | 77 |
| small ResNet + CBAM, attention pool | 66 | 69 | 43 | 70 |
