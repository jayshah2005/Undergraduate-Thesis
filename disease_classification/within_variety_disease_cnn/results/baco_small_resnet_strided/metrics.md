# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| small ResNet (no max pool), mean pool | 0.506 [0.467, 0.545] | 0.520 [0.480, 0.560] | 0.402 [0.355, 0.449] | 0.610 [0.546, 0.667] | 0.439 [0.392, 0.490] | 0.420 [0.374, 0.465] |
| small ResNet (no max pool), attention pool | 0.605 [0.571, 0.638] | 0.601 [0.569, 0.633] | 0.636 [0.579, 0.692] | 0.574 [0.532, 0.610] | 0.531 [0.500, 0.562] | 0.579 [0.540, 0.616] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| small ResNet (no max pool), mean pool | 43 | 64 | 55 | 86 |
| small ResNet (no max pool), attention pool | 68 | 39 | 60 | 81 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| small ResNet (no max pool), mean pool | 0.533 [0.487, 0.577] | 0.524 [0.480, 0.569] | 0.437 [0.378, 0.504] | 0.628 [0.566, 0.690] | 0.584 [0.529, 0.637] | 0.500 [0.443, 0.556] |
| small ResNet (no max pool), attention pool | 0.518 [0.485, 0.549] | 0.500 [0.468, 0.532] | 0.319 [0.281, 0.363] | 0.717 [0.664, 0.770] | 0.573 [0.520, 0.628] | 0.410 [0.365, 0.452] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| small ResNet (no max pool), mean pool | 59 | 76 | 42 | 71 |
| small ResNet (no max pool), attention pool | 43 | 92 | 32 | 81 |
