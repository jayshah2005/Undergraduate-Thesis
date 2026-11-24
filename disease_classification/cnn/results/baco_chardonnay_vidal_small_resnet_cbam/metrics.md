# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| small ResNet + CBAM, mean pool | 0.607 [0.585, 0.631] | 0.608 [0.586, 0.632] | 0.624 [0.591, 0.659] | 0.591 [0.560, 0.619] | 0.632 [0.611, 0.653] | 0.628 [0.602, 0.653] |
| small ResNet + CBAM, attention pool | 0.623 [0.597, 0.648] | 0.622 [0.595, 0.647] | 0.598 [0.561, 0.636] | 0.648 [0.614, 0.679] | 0.657 [0.630, 0.682] | 0.626 [0.597, 0.654] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| small ResNet + CBAM, mean pool | 247 | 149 | 144 | 208 |
| small ResNet + CBAM, attention pool | 237 | 159 | 124 | 228 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| small ResNet + CBAM, mean pool | Baco | 0.481 | 0.492 | 0.607 | 0.354 | 0.529 | 0.566 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.564 | 0.664 | 0.227 | 0.901 | 0.556 | 0.323 |
| small ResNet + CBAM, mean pool | Vidal | 0.562 | 0.668 | 0.838 | 0.286 | 0.725 | 0.777 |
| small ResNet + CBAM, attention pool | Baco | 0.529 | 0.532 | 0.563 | 0.496 | 0.571 | 0.567 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.579 | 0.680 | 0.239 | 0.920 | 0.618 | 0.344 |
| small ResNet + CBAM, attention pool | Vidal | 0.554 | 0.652 | 0.809 | 0.299 | 0.722 | 0.763 |
