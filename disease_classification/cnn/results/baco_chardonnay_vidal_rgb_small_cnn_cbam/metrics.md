# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| CNN + CBAM, mean pool | 0.599 [0.573, 0.625] | 0.598 [0.572, 0.623] | 0.581 [0.540, 0.619] | 0.616 [0.580, 0.651] | 0.630 [0.605, 0.656] | 0.604 [0.574, 0.633] |
| CNN + CBAM, attention pool | 0.608 [0.582, 0.634] | 0.607 [0.582, 0.634] | 0.593 [0.556, 0.629] | 0.622 [0.585, 0.656] | 0.639 [0.613, 0.665] | 0.615 [0.585, 0.644] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| CNN + CBAM, mean pool | 230 | 166 | 135 | 217 |
| CNN + CBAM, attention pool | 235 | 161 | 133 | 219 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| CNN + CBAM, mean pool | Baco | 0.482 | 0.488 | 0.548 | 0.416 | 0.529 | 0.538 |
| CNN + CBAM, mean pool | Chardonnay | 0.571 | 0.676 | 0.216 | 0.926 | 0.613 | 0.319 |
| CNN + CBAM, mean pool | Vidal | 0.526 | 0.628 | 0.792 | 0.260 | 0.706 | 0.747 |
| CNN + CBAM, attention pool | Baco | 0.468 | 0.468 | 0.467 | 0.469 | 0.512 | 0.488 |
| CNN + CBAM, attention pool | Chardonnay | 0.594 | 0.692 | 0.261 | 0.926 | 0.657 | 0.374 |
| CNN + CBAM, attention pool | Vidal | 0.535 | 0.660 | 0.861 | 0.208 | 0.710 | 0.778 |
