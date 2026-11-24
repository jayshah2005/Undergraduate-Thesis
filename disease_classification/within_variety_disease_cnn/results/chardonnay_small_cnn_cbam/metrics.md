# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| CNN + CBAM, mean pool | 0.529 [0.493, 0.562] | 0.532 [0.496, 0.564] | 0.410 [0.352, 0.459] | 0.648 [0.609, 0.688] | 0.526 [0.480, 0.567] | 0.461 [0.411, 0.507] |
| CNN + CBAM, attention pool | 0.565 [0.533, 0.601] | 0.564 [0.532, 0.600] | 0.607 [0.557, 0.656] | 0.523 [0.477, 0.570] | 0.548 [0.519, 0.581] | 0.576 [0.540, 0.611] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| CNN + CBAM, mean pool | 50 | 72 | 45 | 83 |
| CNN + CBAM, attention pool | 74 | 48 | 61 | 67 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| CNN + CBAM, mean pool | 0.594 [0.541, 0.646] | 0.628 [0.580, 0.676] | 0.477 [0.386, 0.568] | 0.710 [0.648, 0.766] | 0.472 [0.404, 0.542] | 0.475 [0.400, 0.544] |
| CNN + CBAM, attention pool | 0.566 [0.514, 0.619] | 0.616 [0.568, 0.668] | 0.398 [0.307, 0.489] | 0.735 [0.673, 0.796] | 0.449 [0.372, 0.531] | 0.422 [0.344, 0.500] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| CNN + CBAM, mean pool | 42 | 46 | 47 | 115 |
| CNN + CBAM, attention pool | 35 | 53 | 43 | 119 |
