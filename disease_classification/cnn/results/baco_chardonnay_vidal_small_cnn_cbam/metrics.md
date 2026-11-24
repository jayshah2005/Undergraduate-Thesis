# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| CNN + CBAM, mean pool | 0.623 [0.602, 0.644] | 0.624 [0.603, 0.646] | 0.644 [0.614, 0.677] | 0.602 [0.574, 0.631] | 0.646 [0.626, 0.665] | 0.645 [0.621, 0.668] |
| CNN + CBAM, attention pool | 0.589 [0.565, 0.612] | 0.592 [0.568, 0.616] | 0.649 [0.614, 0.687] | 0.528 [0.497, 0.560] | 0.608 [0.587, 0.627] | 0.628 [0.603, 0.653] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| CNN + CBAM, mean pool | 255 | 141 | 140 | 212 |
| CNN + CBAM, attention pool | 257 | 139 | 166 | 186 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| CNN + CBAM, mean pool | Baco | 0.527 | 0.540 | 0.674 | 0.381 | 0.565 | 0.615 |
| CNN + CBAM, mean pool | Chardonnay | 0.560 | 0.668 | 0.193 | 0.926 | 0.586 | 0.291 |
| CNN + CBAM, mean pool | Vidal | 0.548 | 0.664 | 0.850 | 0.247 | 0.717 | 0.778 |
| CNN + CBAM, attention pool | Baco | 0.463 | 0.492 | 0.793 | 0.133 | 0.522 | 0.629 |
| CNN + CBAM, attention pool | Chardonnay | 0.548 | 0.660 | 0.170 | 0.926 | 0.556 | 0.261 |
| CNN + CBAM, attention pool | Vidal | 0.527 | 0.624 | 0.780 | 0.273 | 0.707 | 0.742 |
