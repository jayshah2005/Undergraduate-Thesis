# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | 0.584 [0.559, 0.610] | 0.678 [0.635, 0.722] | 0.535 [0.505, 0.566] | 0.430 [0.408, 0.453] | 0.527 [0.499, 0.554] |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | 0.557 [0.532, 0.583] | 0.624 [0.576, 0.671] | 0.523 [0.493, 0.552] | 0.404 [0.380, 0.428] | 0.490 [0.461, 0.519] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| ResNet-18, mean pool | 173 | 82 | 229 | 264 |
| ResNet-18, attention pool | 159 | 96 | 235 | 258 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| ResNet-18, mean pool | Baco | 0.548 | 0.524 | 0.720 | 0.376 | 0.467 | 0.566 |
| ResNet-18, mean pool | Chardonnay | 0.502 | 0.496 | 0.746 | 0.258 | 0.489 | 0.591 |
| ResNet-18, mean pool | Vidal | 0.493 | 0.732 | 0.192 | 0.795 | 0.098 | 0.130 |
| ResNet-18, attention pool | Baco | 0.507 | 0.492 | 0.617 | 0.397 | 0.437 | 0.512 |
| ResNet-18, attention pool | Chardonnay | 0.419 | 0.412 | 0.721 | 0.117 | 0.438 | 0.545 |
| ResNet-18, attention pool | Vidal | 0.514 | 0.768 | 0.192 | 0.835 | 0.119 | 0.147 |

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| ResNet-18, mean pool | 0.602 [0.580, 0.623] | 0.602 [0.579, 0.623] | 0.601 [0.568, 0.634] | 0.602 [0.574, 0.631] | 0.630 [0.608, 0.650] | 0.615 [0.590, 0.639] |
| ResNet-18, attention pool | 0.591 [0.567, 0.615] | 0.590 [0.565, 0.614] | 0.563 [0.528, 0.601] | 0.619 [0.585, 0.651] | 0.625 [0.599, 0.649] | 0.592 [0.565, 0.621] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| ResNet-18, mean pool | 238 | 158 | 140 | 212 |
| ResNet-18, attention pool | 223 | 173 | 134 | 218 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| ResNet-18, mean pool | Baco | 0.501 | 0.516 | 0.667 | 0.336 | 0.545 | 0.600 |
| ResNet-18, mean pool | Chardonnay | 0.535 | 0.660 | 0.114 | 0.957 | 0.588 | 0.190 |
| ResNet-18, mean pool | Vidal | 0.522 | 0.628 | 0.798 | 0.247 | 0.704 | 0.748 |
| ResNet-18, attention pool | Baco | 0.489 | 0.496 | 0.563 | 0.416 | 0.535 | 0.549 |
| ResNet-18, attention pool | Chardonnay | 0.542 | 0.652 | 0.170 | 0.914 | 0.517 | 0.256 |
| ResNet-18, attention pool | Vidal | 0.531 | 0.620 | 0.763 | 0.299 | 0.710 | 0.735 |
