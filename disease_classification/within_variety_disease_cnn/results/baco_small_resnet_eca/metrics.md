# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| small ResNet + ECA, mean pool | 0.491 [0.464, 0.522] | 0.460 [0.435, 0.488] | 0.720 [0.673, 0.766] | 0.262 [0.234, 0.298] | 0.425 [0.407, 0.446] | 0.535 [0.509, 0.562] |
| small ResNet + ECA, attention pool | 0.586 [0.552, 0.617] | 0.597 [0.565, 0.625] | 0.505 [0.449, 0.561] | 0.667 [0.631, 0.695] | 0.535 [0.495, 0.570] | 0.519 [0.473, 0.561] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| small ResNet + ECA, mean pool | 77 | 30 | 104 | 37 |
| small ResNet + ECA, attention pool | 54 | 53 | 47 | 94 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| small ResNet + ECA, mean pool | 0.531 [0.487, 0.579] | 0.524 [0.480, 0.573] | 0.452 [0.393, 0.511] | 0.611 [0.540, 0.681] | 0.581 [0.530, 0.640] | 0.508 [0.455, 0.563] |
| small ResNet + ECA, attention pool | 0.537 [0.494, 0.579] | 0.524 [0.484, 0.569] | 0.393 [0.333, 0.452] | 0.681 [0.619, 0.743] | 0.596 [0.537, 0.652] | 0.473 [0.416, 0.529] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| small ResNet + ECA, mean pool | 61 | 74 | 44 | 69 |
| small ResNet + ECA, attention pool | 53 | 82 | 36 | 77 |
