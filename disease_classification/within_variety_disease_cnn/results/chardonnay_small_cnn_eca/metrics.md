# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| CNN + ECA, mean pool | 0.573 [0.545, 0.601] | 0.572 [0.544, 0.600] | 0.615 [0.574, 0.656] | 0.531 [0.492, 0.578] | 0.556 [0.529, 0.583] | 0.584 [0.551, 0.613] |
| CNN + ECA, attention pool | 0.580 [0.548, 0.608] | 0.580 [0.548, 0.608] | 0.574 [0.533, 0.615] | 0.586 [0.547, 0.625] | 0.569 [0.537, 0.598] | 0.571 [0.539, 0.603] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| CNN + ECA, mean pool | 75 | 47 | 60 | 68 |
| CNN + ECA, attention pool | 70 | 52 | 53 | 75 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| CNN + ECA, mean pool | 0.536 [0.491, 0.580] | 0.520 [0.476, 0.560] | 0.591 [0.523, 0.659] | 0.481 [0.426, 0.531] | 0.382 [0.345, 0.419] | 0.464 [0.418, 0.509] |
| CNN + ECA, attention pool | 0.578 [0.523, 0.632] | 0.628 [0.576, 0.676] | 0.409 [0.318, 0.500] | 0.747 [0.691, 0.802] | 0.468 [0.386, 0.548] | 0.436 [0.350, 0.512] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| CNN + ECA, mean pool | 52 | 36 | 84 | 78 |
| CNN + ECA, attention pool | 36 | 52 | 41 | 121 |
