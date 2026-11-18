# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | 0.580 [0.560, 0.599] | 0.643 [0.612, 0.671] | 0.548 [0.521, 0.572] | 0.424 [0.405, 0.441] | 0.511 [0.490, 0.530] |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | 0.563 [0.544, 0.580] | 0.725 [0.694, 0.757] | 0.479 [0.456, 0.499] | 0.419 [0.404, 0.433] | 0.531 [0.512, 0.549] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| small ResNet (no max pool), mean pool | 164 | 91 | 223 | 270 |
| small ResNet (no max pool), attention pool | 185 | 70 | 257 | 236 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| small ResNet (no max pool), mean pool | Baco | 0.480 | 0.448 | 0.720 | 0.241 | 0.418 | 0.529 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.497 | 0.492 | 0.705 | 0.289 | 0.486 | 0.575 |
| small ResNet (no max pool), mean pool | Vidal | 0.463 | 0.800 | 0.038 | 0.888 | 0.038 | 0.038 |
| small ResNet (no max pool), attention pool | Baco | 0.524 | 0.484 | 0.813 | 0.234 | 0.446 | 0.576 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.475 | 0.468 | 0.762 | 0.188 | 0.472 | 0.583 |
| small ResNet (no max pool), attention pool | Vidal | 0.496 | 0.736 | 0.192 | 0.799 | 0.100 | 0.132 |

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| small ResNet (no max pool), mean pool | 0.624 [0.599, 0.646] | 0.627 [0.603, 0.650] | 0.682 [0.646, 0.715] | 0.565 [0.531, 0.597] | 0.638 [0.616, 0.659] | 0.659 [0.635, 0.683] |
| small ResNet (no max pool), attention pool | 0.590 [0.566, 0.614] | 0.592 [0.568, 0.616] | 0.624 [0.591, 0.662] | 0.557 [0.526, 0.588] | 0.613 [0.591, 0.634] | 0.618 [0.593, 0.644] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| small ResNet (no max pool), mean pool | 270 | 126 | 153 | 199 |
| small ResNet (no max pool), attention pool | 247 | 149 | 156 | 196 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| small ResNet (no max pool), mean pool | Baco | 0.530 | 0.548 | 0.733 | 0.327 | 0.566 | 0.639 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.565 | 0.668 | 0.216 | 0.914 | 0.576 | 0.314 |
| small ResNet (no max pool), mean pool | Vidal | 0.530 | 0.664 | 0.879 | 0.182 | 0.707 | 0.784 |
| small ResNet (no max pool), attention pool | Baco | 0.451 | 0.476 | 0.726 | 0.177 | 0.513 | 0.601 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.527 | 0.656 | 0.091 | 0.963 | 0.571 | 0.157 |
| small ResNet (no max pool), attention pool | Vidal | 0.537 | 0.644 | 0.815 | 0.260 | 0.712 | 0.760 |
