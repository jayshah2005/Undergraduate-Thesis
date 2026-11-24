# Disease: whole-leaf CNN on Chardonnay, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.588 [0.536, 0.644] | 0.592 [0.540, 0.648] | 0.410 [0.328, 0.492] | 0.766 [0.695, 0.828] | 0.625 [0.544, 0.711] | 0.495 [0.413, 0.574] |
| CNN + CBAM, mean pool | 0.528 [0.478, 0.580] | 0.532 [0.480, 0.584] | 0.369 [0.303, 0.443] | 0.688 [0.617, 0.758] | 0.529 [0.458, 0.605] | 0.435 [0.365, 0.505] |
| CNN + CBAM, attention pool | 0.510 [0.467, 0.554] | 0.516 [0.472, 0.560] | 0.279 [0.221, 0.336] | 0.742 [0.680, 0.805] | 0.507 [0.429, 0.594] | 0.360 [0.295, 0.419] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 50 | 72 | 30 | 98 |
| CNN + CBAM, mean pool | 45 | 77 | 40 | 88 |
| CNN + CBAM, attention pool | 34 | 88 | 33 | 95 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.619 [0.557, 0.679] | 0.648 [0.592, 0.704] | 0.523 [0.420, 0.625] | 0.716 [0.654, 0.784] | 0.500 [0.424, 0.582] | 0.511 [0.426, 0.587] |
| CNN + CBAM, mean pool | 0.588 [0.534, 0.644] | 0.624 [0.572, 0.676] | 0.466 [0.375, 0.557] | 0.710 [0.648, 0.778] | 0.466 [0.395, 0.544] | 0.466 [0.391, 0.541] |
| CNN + CBAM, attention pool | 0.563 [0.504, 0.620] | 0.608 [0.556, 0.660] | 0.409 [0.318, 0.511] | 0.716 [0.654, 0.778] | 0.439 [0.357, 0.521] | 0.424 [0.337, 0.506] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 46 | 42 | 46 | 116 |
| CNN + CBAM, mean pool | 41 | 47 | 47 | 115 |
| CNN + CBAM, attention pool | 36 | 52 | 46 | 116 |
