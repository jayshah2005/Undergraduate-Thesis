# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| CNN, mean pool | 0.531 [0.487, 0.575] | 0.532 [0.488, 0.576] | 0.484 [0.426, 0.541] | 0.578 [0.516, 0.641] | 0.522 [0.475, 0.571] | 0.502 [0.451, 0.552] |
| CNN, attention pool | 0.569 [0.525, 0.613] | 0.572 [0.528, 0.616] | 0.426 [0.361, 0.492] | 0.711 [0.648, 0.773] | 0.584 [0.521, 0.651] | 0.493 [0.433, 0.550] |
| small ResNet, mean pool | 0.580 [0.540, 0.620] | 0.580 [0.540, 0.620] | 0.590 [0.533, 0.639] | 0.570 [0.516, 0.633] | 0.567 [0.527, 0.610] | 0.578 [0.537, 0.619] |
| small ResNet, attention pool | 0.562 [0.520, 0.602] | 0.564 [0.524, 0.604] | 0.467 [0.402, 0.525] | 0.656 [0.602, 0.711] | 0.564 [0.514, 0.615] | 0.511 [0.457, 0.560] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| CNN, mean pool | 59 | 63 | 54 | 74 |
| CNN, attention pool | 52 | 70 | 37 | 91 |
| small ResNet, mean pool | 72 | 50 | 55 | 73 |
| small ResNet, attention pool | 57 | 65 | 44 | 84 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| CNN, mean pool | 0.584 [0.529, 0.642] | 0.636 [0.584, 0.688] | 0.409 [0.307, 0.511] | 0.759 [0.698, 0.821] | 0.480 [0.397, 0.570] | 0.442 [0.352, 0.526] |
| CNN, attention pool | 0.583 [0.525, 0.641] | 0.624 [0.572, 0.680] | 0.443 [0.352, 0.545] | 0.722 [0.660, 0.784] | 0.464 [0.385, 0.548] | 0.453 [0.371, 0.533] |
| small ResNet, mean pool | 0.578 [0.524, 0.631] | 0.628 [0.576, 0.680] | 0.409 [0.318, 0.500] | 0.747 [0.685, 0.809] | 0.468 [0.388, 0.556] | 0.436 [0.352, 0.512] |
| small ResNet, attention pool | 0.584 [0.529, 0.640] | 0.636 [0.584, 0.688] | 0.409 [0.318, 0.500] | 0.759 [0.698, 0.827] | 0.480 [0.395, 0.573] | 0.442 [0.356, 0.523] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| CNN, mean pool | 36 | 52 | 39 | 123 |
| CNN, attention pool | 39 | 49 | 45 | 117 |
| small ResNet, mean pool | 36 | 52 | 41 | 121 |
| small ResNet, attention pool | 36 | 52 | 39 | 123 |
