# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet, mean pool | 0.585 [0.556, 0.614] | 0.598 [0.570, 0.624] | 0.547 [0.500, 0.594] | 0.623 [0.591, 0.656] | 0.428 [0.398, 0.458] | 0.480 [0.445, 0.514] |
| small ResNet, attention pool | 0.564 [0.535, 0.590] | 0.563 [0.536, 0.588] | 0.567 [0.520, 0.610] | 0.561 [0.530, 0.591] | 0.399 [0.372, 0.424] | 0.468 [0.436, 0.498] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| small ResNet, mean pool | 139 | 115 | 186 | 308 |
| small ResNet, attention pool | 144 | 110 | 217 | 277 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |
| small ResNet, mean pool | Baco | 0.495 | 0.500 | 0.458 | 0.532 | 0.426 | 0.441 |
| small ResNet, mean pool | Chardonnay | 0.474 | 0.468 | 0.721 | 0.227 | 0.471 | 0.570 |
| small ResNet, mean pool | Vidal | 0.493 | 0.824 | 0.080 | 0.907 | 0.087 | 0.083 |
| small ResNet, attention pool | Baco | 0.483 | 0.488 | 0.449 | 0.518 | 0.414 | 0.430 |
| small ResNet, attention pool | Chardonnay | 0.412 | 0.404 | 0.762 | 0.062 | 0.437 | 0.555 |
| small ResNet, attention pool | Vidal | 0.496 | 0.796 | 0.120 | 0.871 | 0.094 | 0.105 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |
| small ResNet, mean pool | 0.608 [0.583, 0.633] | 0.606 [0.580, 0.631] | 0.564 [0.526, 0.602] | 0.652 [0.615, 0.685] | 0.641 [0.613, 0.668] | 0.600 [0.570, 0.629] |
| small ResNet, attention pool | 0.598 [0.572, 0.625] | 0.598 [0.572, 0.624] | 0.587 [0.551, 0.622] | 0.610 [0.573, 0.646] | 0.623 [0.597, 0.651] | 0.604 [0.576, 0.633] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |
| small ResNet, mean pool | 221 | 171 | 124 | 232 |
| small ResNet, attention pool | 230 | 162 | 139 | 217 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |
| small ResNet, mean pool | Baco | 0.473 | 0.468 | 0.415 | 0.531 | 0.514 | 0.459 |
| small ResNet, mean pool | Chardonnay | 0.583 | 0.688 | 0.227 | 0.938 | 0.667 | 0.339 |
| small ResNet, mean pool | Vidal | 0.552 | 0.660 | 0.858 | 0.247 | 0.704 | 0.773 |
| small ResNet, attention pool | Baco | 0.464 | 0.464 | 0.459 | 0.469 | 0.508 | 0.482 |
| small ResNet, attention pool | Chardonnay | 0.567 | 0.668 | 0.227 | 0.907 | 0.571 | 0.325 |
| small ResNet, attention pool | Vidal | 0.543 | 0.660 | 0.876 | 0.210 | 0.698 | 0.777 |
