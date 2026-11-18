# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | 0.607 [0.582, 0.632] | 0.557 [0.510, 0.600] | 0.633 [0.602, 0.665] | 0.440 [0.411, 0.468] | 0.491 [0.459, 0.523] |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | 0.567 [0.539, 0.594] | 0.557 [0.510, 0.600] | 0.572 [0.540, 0.602] | 0.402 [0.374, 0.429] | 0.467 [0.433, 0.499] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| small ResNet, mean pool | 142 | 113 | 181 | 312 |
| small ResNet, attention pool | 142 | 113 | 211 | 282 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| small ResNet, mean pool | Baco | 0.524 | 0.532 | 0.467 | 0.582 | 0.459 | 0.463 |
| small ResNet, mean pool | Chardonnay | 0.525 | 0.520 | 0.746 | 0.305 | 0.506 | 0.603 |
| small ResNet, mean pool | Vidal | 0.446 | 0.768 | 0.038 | 0.853 | 0.029 | 0.033 |
| small ResNet, attention pool | Baco | 0.477 | 0.484 | 0.430 | 0.525 | 0.407 | 0.418 |
| small ResNet, attention pool | Chardonnay | 0.440 | 0.432 | 0.762 | 0.117 | 0.451 | 0.567 |
| small ResNet, attention pool | Vidal | 0.488 | 0.784 | 0.115 | 0.862 | 0.088 | 0.100 |

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| small ResNet, mean pool | 0.604 [0.579, 0.630] | 0.602 [0.575, 0.627] | 0.556 [0.515, 0.593] | 0.653 [0.619, 0.688] | 0.643 [0.615, 0.671] | 0.596 [0.565, 0.625] |
| small ResNet, attention pool | 0.596 [0.568, 0.622] | 0.594 [0.566, 0.620] | 0.558 [0.518, 0.598] | 0.634 [0.597, 0.670] | 0.631 [0.603, 0.660] | 0.592 [0.561, 0.623] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| small ResNet, mean pool | 220 | 176 | 122 | 230 |
| small ResNet, attention pool | 221 | 175 | 129 | 223 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| small ResNet, mean pool | Baco | 0.477 | 0.472 | 0.415 | 0.540 | 0.519 | 0.461 |
| small ResNet, mean pool | Chardonnay | 0.568 | 0.676 | 0.205 | 0.932 | 0.621 | 0.308 |
| small ResNet, mean pool | Vidal | 0.539 | 0.656 | 0.844 | 0.234 | 0.712 | 0.772 |
| small ResNet, attention pool | Baco | 0.480 | 0.476 | 0.437 | 0.522 | 0.522 | 0.476 |
| small ResNet, attention pool | Chardonnay | 0.559 | 0.660 | 0.216 | 0.901 | 0.543 | 0.309 |
| small ResNet, attention pool | Vidal | 0.530 | 0.644 | 0.827 | 0.234 | 0.708 | 0.763 |
