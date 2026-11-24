# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| small ResNet + ECA, mean pool | 0.554 [0.522, 0.587] | 0.556 [0.524, 0.588] | 0.492 [0.434, 0.541] | 0.617 [0.570, 0.664] | 0.550 [0.513, 0.590] | 0.519 [0.476, 0.561] |
| small ResNet + ECA, attention pool | 0.533 [0.498, 0.566] | 0.532 [0.496, 0.564] | 0.590 [0.541, 0.631] | 0.477 [0.430, 0.523] | 0.518 [0.486, 0.548] | 0.552 [0.517, 0.583] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| small ResNet + ECA, mean pool | 60 | 62 | 49 | 79 |
| small ResNet + ECA, attention pool | 72 | 50 | 67 | 61 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| small ResNet + ECA, mean pool | 0.577 [0.526, 0.627] | 0.600 [0.552, 0.648] | 0.500 [0.420, 0.580] | 0.654 [0.593, 0.716] | 0.440 [0.381, 0.500] | 0.468 [0.402, 0.530] |
| small ResNet + ECA, attention pool | 0.608 [0.556, 0.659] | 0.680 [0.636, 0.724] | 0.364 [0.273, 0.455] | 0.852 [0.802, 0.901] | 0.571 [0.469, 0.673] | 0.444 [0.347, 0.532] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| small ResNet + ECA, mean pool | 44 | 44 | 56 | 106 |
| small ResNet + ECA, attention pool | 32 | 56 | 24 | 138 |
