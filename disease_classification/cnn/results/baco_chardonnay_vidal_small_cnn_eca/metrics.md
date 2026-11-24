# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| CNN + ECA, mean pool | 0.580 [0.556, 0.603] | 0.584 [0.560, 0.608] | 0.649 [0.614, 0.687] | 0.511 [0.483, 0.540] | 0.599 [0.578, 0.619] | 0.623 [0.597, 0.649] |
| CNN + ECA, attention pool | 0.609 [0.589, 0.627] | 0.612 [0.592, 0.631] | 0.667 [0.639, 0.697] | 0.551 [0.526, 0.574] | 0.626 [0.608, 0.642] | 0.645 [0.625, 0.665] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| CNN + ECA, mean pool | 257 | 139 | 172 | 180 |
| CNN + ECA, attention pool | 264 | 132 | 158 | 194 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| CNN + ECA, mean pool | Baco | 0.441 | 0.472 | 0.793 | 0.088 | 0.510 | 0.620 |
| CNN + ECA, mean pool | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| CNN + ECA, mean pool | Vidal | 0.530 | 0.624 | 0.775 | 0.286 | 0.709 | 0.740 |
| CNN + ECA, attention pool | Baco | 0.503 | 0.528 | 0.785 | 0.221 | 0.546 | 0.644 |
| CNN + ECA, attention pool | Chardonnay | 0.553 | 0.660 | 0.193 | 0.914 | 0.548 | 0.286 |
| CNN + ECA, attention pool | Vidal | 0.544 | 0.648 | 0.815 | 0.273 | 0.716 | 0.762 |
