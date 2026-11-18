# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | 0.586 [0.564, 0.606] | 0.638 [0.602, 0.673] | 0.559 [0.530, 0.585] | 0.426 [0.407, 0.445] | 0.511 [0.488, 0.533] |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | 0.576 [0.557, 0.596] | 0.701 [0.665, 0.736] | 0.512 [0.490, 0.534] | 0.425 [0.408, 0.442] | 0.529 [0.507, 0.550] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| small ResNet (no max pool), mean pool | 162 | 92 | 218 | 276 |
| small ResNet (no max pool), attention pool | 178 | 76 | 241 | 253 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| small ResNet (no max pool), mean pool | Baco | 0.528 | 0.504 | 0.701 | 0.355 | 0.452 | 0.549 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.458 | 0.452 | 0.689 | 0.227 | 0.459 | 0.551 |
| small ResNet (no max pool), mean pool | Vidal | 0.498 | 0.800 | 0.120 | 0.876 | 0.097 | 0.107 |
| small ResNet (no max pool), attention pool | Baco | 0.538 | 0.512 | 0.729 | 0.348 | 0.459 | 0.563 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.467 | 0.460 | 0.762 | 0.172 | 0.467 | 0.579 |
| small ResNet (no max pool), attention pool | Vidal | 0.544 | 0.756 | 0.280 | 0.809 | 0.140 | 0.187 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |
| small ResNet (no max pool), mean pool | 0.616 [0.592, 0.639] | 0.616 [0.592, 0.640] | 0.628 [0.594, 0.661] | 0.604 [0.570, 0.635] | 0.636 [0.613, 0.659] | 0.632 [0.606, 0.657] |
| small ResNet (no max pool), attention pool | 0.610 [0.589, 0.630] | 0.614 [0.594, 0.634] | 0.694 [0.663, 0.727] | 0.525 [0.497, 0.551] | 0.617 [0.600, 0.634] | 0.653 [0.632, 0.675] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |
| small ResNet (no max pool), mean pool | 246 | 146 | 141 | 215 |
| small ResNet (no max pool), attention pool | 272 | 120 | 169 | 187 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.468 | 0.548 | 0.372 | 0.510 | 0.529 |
| logistic regression | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| logistic regression | Vidal | 0.546 | 0.668 | 0.893 | 0.198 | 0.699 | 0.784 |
| small ResNet (no max pool), mean pool | Baco | 0.538 | 0.548 | 0.659 | 0.416 | 0.574 | 0.614 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.551 | 0.664 | 0.170 | 0.932 | 0.577 | 0.263 |
| small ResNet (no max pool), mean pool | Vidal | 0.525 | 0.636 | 0.840 | 0.210 | 0.689 | 0.757 |
| small ResNet (no max pool), attention pool | Baco | 0.474 | 0.508 | 0.859 | 0.088 | 0.530 | 0.655 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.538 | 0.660 | 0.125 | 0.951 | 0.579 | 0.206 |
| small ResNet (no max pool), attention pool | Vidal | 0.571 | 0.672 | 0.858 | 0.284 | 0.714 | 0.780 |
