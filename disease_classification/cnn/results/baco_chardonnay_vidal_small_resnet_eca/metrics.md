# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| small ResNet + ECA, mean pool | 0.609 [0.586, 0.632] | 0.612 [0.590, 0.635] | 0.669 [0.636, 0.702] | 0.548 [0.517, 0.580] | 0.625 [0.605, 0.645] | 0.646 [0.624, 0.669] |
| small ResNet + ECA, attention pool | 0.626 [0.605, 0.646] | 0.627 [0.606, 0.647] | 0.639 [0.609, 0.667] | 0.614 [0.585, 0.639] | 0.650 [0.630, 0.669] | 0.645 [0.622, 0.666] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| small ResNet + ECA, mean pool | 265 | 131 | 159 | 193 |
| small ResNet + ECA, attention pool | 253 | 143 | 136 | 216 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| small ResNet + ECA, mean pool | Baco | 0.492 | 0.508 | 0.674 | 0.310 | 0.538 | 0.599 |
| small ResNet + ECA, mean pool | Chardonnay | 0.563 | 0.656 | 0.250 | 0.877 | 0.524 | 0.338 |
| small ResNet + ECA, mean pool | Vidal | 0.543 | 0.672 | 0.879 | 0.208 | 0.714 | 0.788 |
| small ResNet + ECA, attention pool | Baco | 0.521 | 0.532 | 0.644 | 0.398 | 0.561 | 0.600 |
| small ResNet + ECA, attention pool | Chardonnay | 0.543 | 0.656 | 0.159 | 0.926 | 0.538 | 0.246 |
| small ResNet + ECA, attention pool | Vidal | 0.576 | 0.692 | 0.879 | 0.273 | 0.731 | 0.798 |
