# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| ResNet-18, mean pool | 0.601 [0.572, 0.627] | 0.582 [0.555, 0.607] | 0.661 [0.614, 0.705] | 0.540 [0.508, 0.571] | 0.425 [0.400, 0.448] | 0.518 [0.488, 0.545] |
| ResNet-18, attention pool | 0.577 [0.549, 0.603] | 0.556 [0.531, 0.582] | 0.642 [0.598, 0.685] | 0.512 [0.482, 0.543] | 0.403 [0.380, 0.425] | 0.495 [0.467, 0.521] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| ResNet-18, mean pool | 168 | 86 | 227 | 267 |
| ResNet-18, attention pool | 163 | 91 | 241 | 253 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| ResNet-18, mean pool | Baco | 0.507 | 0.488 | 0.645 | 0.369 | 0.437 | 0.521 |
| ResNet-18, mean pool | Chardonnay | 0.494 | 0.488 | 0.762 | 0.227 | 0.484 | 0.592 |
| ResNet-18, mean pool | Vidal | 0.533 | 0.768 | 0.240 | 0.827 | 0.133 | 0.171 |
| ResNet-18, attention pool | Baco | 0.522 | 0.504 | 0.654 | 0.390 | 0.449 | 0.532 |
| ResNet-18, attention pool | Chardonnay | 0.443 | 0.436 | 0.730 | 0.156 | 0.452 | 0.558 |
| ResNet-18, attention pool | Vidal | 0.476 | 0.728 | 0.160 | 0.791 | 0.078 | 0.105 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |
| ResNet-18, mean pool | 0.590 [0.567, 0.614] | 0.591 [0.567, 0.615] | 0.605 [0.569, 0.640] | 0.576 [0.545, 0.610] | 0.611 [0.589, 0.634] | 0.608 [0.580, 0.633] |
| ResNet-18, attention pool | 0.577 [0.552, 0.601] | 0.578 [0.553, 0.602] | 0.597 [0.564, 0.633] | 0.556 [0.520, 0.590] | 0.597 [0.574, 0.620] | 0.597 [0.571, 0.623] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |
| ResNet-18, mean pool | 237 | 155 | 151 | 205 |
| ResNet-18, attention pool | 234 | 158 | 158 | 198 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.468 | 0.548 | 0.372 | 0.510 | 0.529 |
| logistic regression | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| logistic regression | Vidal | 0.546 | 0.668 | 0.893 | 0.198 | 0.699 | 0.784 |
| ResNet-18, mean pool | Baco | 0.493 | 0.508 | 0.667 | 0.319 | 0.539 | 0.596 |
| ResNet-18, mean pool | Chardonnay | 0.533 | 0.640 | 0.170 | 0.895 | 0.469 | 0.250 |
| ResNet-18, mean pool | Vidal | 0.539 | 0.624 | 0.781 | 0.296 | 0.698 | 0.737 |
| ResNet-18, attention pool | Baco | 0.469 | 0.472 | 0.496 | 0.442 | 0.515 | 0.506 |
| ResNet-18, attention pool | Chardonnay | 0.526 | 0.608 | 0.250 | 0.802 | 0.407 | 0.310 |
| ResNet-18, attention pool | Vidal | 0.540 | 0.652 | 0.858 | 0.222 | 0.697 | 0.769 |
