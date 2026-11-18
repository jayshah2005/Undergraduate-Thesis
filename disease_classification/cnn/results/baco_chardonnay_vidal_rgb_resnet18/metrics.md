# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| ResNet-18, mean pool | 0.577 [0.551, 0.600] | 0.551 [0.525, 0.575] | 0.657 [0.618, 0.697] | 0.496 [0.464, 0.526] | 0.401 [0.381, 0.421] | 0.499 [0.473, 0.522] |
| ResNet-18, attention pool | 0.596 [0.568, 0.623] | 0.575 [0.548, 0.600] | 0.661 [0.618, 0.705] | 0.530 [0.496, 0.563] | 0.420 [0.397, 0.443] | 0.514 [0.485, 0.542] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| ResNet-18, mean pool | 167 | 87 | 249 | 245 |
| ResNet-18, attention pool | 168 | 86 | 232 | 262 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |
| ResNet-18, mean pool | Baco | 0.498 | 0.476 | 0.664 | 0.333 | 0.430 | 0.522 |
| ResNet-18, mean pool | Chardonnay | 0.463 | 0.456 | 0.746 | 0.180 | 0.464 | 0.572 |
| ResNet-18, mean pool | Vidal | 0.489 | 0.720 | 0.200 | 0.778 | 0.091 | 0.125 |
| ResNet-18, attention pool | Baco | 0.530 | 0.512 | 0.664 | 0.397 | 0.455 | 0.540 |
| ResNet-18, attention pool | Chardonnay | 0.463 | 0.456 | 0.762 | 0.164 | 0.465 | 0.578 |
| ResNet-18, attention pool | Vidal | 0.491 | 0.756 | 0.160 | 0.822 | 0.091 | 0.116 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |
| ResNet-18, mean pool | 0.562 [0.536, 0.589] | 0.560 [0.533, 0.587] | 0.520 [0.485, 0.559] | 0.604 [0.567, 0.640] | 0.591 [0.563, 0.620] | 0.554 [0.523, 0.582] |
| ResNet-18, attention pool | 0.587 [0.562, 0.613] | 0.590 [0.564, 0.616] | 0.643 [0.607, 0.681] | 0.531 [0.494, 0.567] | 0.601 [0.579, 0.626] | 0.621 [0.595, 0.648] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |
| ResNet-18, mean pool | 204 | 188 | 141 | 215 |
| ResNet-18, attention pool | 252 | 140 | 167 | 189 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |
| ResNet-18, mean pool | Baco | 0.493 | 0.496 | 0.526 | 0.460 | 0.538 | 0.532 |
| ResNet-18, mean pool | Chardonnay | 0.494 | 0.596 | 0.148 | 0.840 | 0.333 | 0.205 |
| ResNet-18, mean pool | Vidal | 0.522 | 0.588 | 0.710 | 0.333 | 0.690 | 0.700 |
| ResNet-18, attention pool | Baco | 0.527 | 0.544 | 0.726 | 0.327 | 0.563 | 0.634 |
| ResNet-18, attention pool | Chardonnay | 0.514 | 0.596 | 0.239 | 0.790 | 0.382 | 0.294 |
| ResNet-18, attention pool | Vidal | 0.542 | 0.628 | 0.787 | 0.296 | 0.700 | 0.741 |
