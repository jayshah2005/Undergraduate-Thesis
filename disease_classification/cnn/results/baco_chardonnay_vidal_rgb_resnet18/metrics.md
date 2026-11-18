# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | 0.584 [0.557, 0.611] | 0.592 [0.549, 0.635] | 0.580 [0.548, 0.613] | 0.422 [0.396, 0.448] | 0.493 [0.462, 0.523] |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | 0.594 [0.567, 0.620] | 0.663 [0.620, 0.706] | 0.558 [0.525, 0.590] | 0.437 [0.412, 0.461] | 0.526 [0.497, 0.554] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| ResNet-18, mean pool | 151 | 104 | 207 | 286 |
| ResNet-18, attention pool | 169 | 86 | 218 | 275 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| ResNet-18, mean pool | Baco | 0.523 | 0.516 | 0.570 | 0.475 | 0.452 | 0.504 |
| ResNet-18, mean pool | Chardonnay | 0.509 | 0.504 | 0.713 | 0.305 | 0.494 | 0.584 |
| ResNet-18, mean pool | Vidal | 0.459 | 0.732 | 0.115 | 0.804 | 0.064 | 0.082 |
| ResNet-18, attention pool | Baco | 0.549 | 0.536 | 0.645 | 0.454 | 0.473 | 0.545 |
| ResNet-18, attention pool | Chardonnay | 0.471 | 0.464 | 0.779 | 0.164 | 0.470 | 0.586 |
| ResNet-18, attention pool | Vidal | 0.520 | 0.780 | 0.192 | 0.848 | 0.128 | 0.154 |

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| ResNet-18, mean pool | 0.605 [0.578, 0.632] | 0.604 [0.578, 0.631] | 0.588 [0.553, 0.626] | 0.622 [0.585, 0.659] | 0.637 [0.609, 0.664] | 0.612 [0.584, 0.641] |
| ResNet-18, attention pool | 0.581 [0.557, 0.605] | 0.576 [0.552, 0.600] | 0.505 [0.472, 0.543] | 0.656 [0.625, 0.688] | 0.623 [0.596, 0.651] | 0.558 [0.529, 0.588] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| ResNet-18, mean pool | 233 | 163 | 133 | 219 |
| ResNet-18, attention pool | 200 | 196 | 121 | 231 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| ResNet-18, mean pool | Baco | 0.511 | 0.512 | 0.526 | 0.496 | 0.555 | 0.540 |
| ResNet-18, mean pool | Chardonnay | 0.546 | 0.644 | 0.216 | 0.877 | 0.487 | 0.299 |
| ResNet-18, mean pool | Vidal | 0.550 | 0.656 | 0.827 | 0.273 | 0.719 | 0.769 |
| ResNet-18, attention pool | Baco | 0.487 | 0.488 | 0.496 | 0.478 | 0.532 | 0.513 |
| ResNet-18, attention pool | Chardonnay | 0.528 | 0.644 | 0.136 | 0.920 | 0.480 | 0.212 |
| ResNet-18, attention pool | Vidal | 0.532 | 0.596 | 0.699 | 0.364 | 0.712 | 0.706 |
