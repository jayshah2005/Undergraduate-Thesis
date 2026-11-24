# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| CNN + CBAM, mean pool | 0.575 [0.541, 0.608] | 0.560 [0.528, 0.597] | 0.682 [0.636, 0.720] | 0.468 [0.418, 0.518] | 0.493 [0.465, 0.524] | 0.573 [0.541, 0.603] |
| CNN + CBAM, attention pool | 0.538 [0.501, 0.578] | 0.552 [0.516, 0.593] | 0.430 [0.374, 0.486] | 0.645 [0.596, 0.702] | 0.479 [0.433, 0.533] | 0.453 [0.406, 0.502] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| CNN + CBAM, mean pool | 73 | 34 | 75 | 66 |
| CNN + CBAM, attention pool | 46 | 61 | 50 | 91 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| CNN + CBAM, mean pool | 0.486 [0.449, 0.521] | 0.468 [0.431, 0.504] | 0.281 [0.222, 0.341] | 0.690 [0.646, 0.735] | 0.521 [0.453, 0.579] | 0.365 [0.302, 0.425] |
| CNN + CBAM, attention pool | 0.474 [0.427, 0.522] | 0.468 [0.419, 0.516] | 0.400 [0.333, 0.474] | 0.549 [0.478, 0.619] | 0.514 [0.457, 0.571] | 0.450 [0.391, 0.510] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| CNN + CBAM, mean pool | 38 | 97 | 35 | 78 |
| CNN + CBAM, attention pool | 54 | 81 | 51 | 62 |
