# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN, mean pool | 0.565 [0.535, 0.593] | 0.560 [0.533, 0.587] | 0.579 [0.531, 0.626] | 0.551 [0.518, 0.581] | 0.398 [0.372, 0.424] | 0.472 [0.439, 0.503] |
| CNN, attention pool | 0.580 [0.552, 0.607] | 0.591 [0.564, 0.616] | 0.547 [0.500, 0.591] | 0.613 [0.583, 0.642] | 0.421 [0.392, 0.449] | 0.476 [0.443, 0.508] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| CNN, mean pool | 147 | 107 | 222 | 272 |
| CNN, attention pool | 139 | 115 | 191 | 303 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |
| CNN, mean pool | Baco | 0.462 | 0.460 | 0.477 | 0.447 | 0.395 | 0.432 |
| CNN, mean pool | Chardonnay | 0.436 | 0.428 | 0.762 | 0.109 | 0.449 | 0.565 |
| CNN, mean pool | Vidal | 0.493 | 0.792 | 0.120 | 0.867 | 0.091 | 0.103 |
| CNN, attention pool | Baco | 0.540 | 0.548 | 0.477 | 0.603 | 0.477 | 0.477 |
| CNN, attention pool | Chardonnay | 0.423 | 0.416 | 0.697 | 0.148 | 0.438 | 0.538 |
| CNN, attention pool | Vidal | 0.502 | 0.808 | 0.120 | 0.884 | 0.103 | 0.111 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |
| CNN, mean pool | 0.602 [0.576, 0.628] | 0.600 [0.575, 0.627] | 0.569 [0.531, 0.607] | 0.635 [0.601, 0.669] | 0.632 [0.605, 0.659] | 0.599 [0.569, 0.628] |
| CNN, attention pool | 0.607 [0.582, 0.633] | 0.606 [0.580, 0.631] | 0.569 [0.531, 0.605] | 0.646 [0.612, 0.680] | 0.639 [0.611, 0.666] | 0.602 [0.573, 0.631] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |
| CNN, mean pool | 223 | 169 | 130 | 226 |
| CNN, attention pool | 223 | 169 | 126 | 230 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |
| CNN, mean pool | Baco | 0.455 | 0.452 | 0.415 | 0.496 | 0.496 | 0.452 |
| CNN, mean pool | Chardonnay | 0.591 | 0.692 | 0.250 | 0.932 | 0.667 | 0.364 |
| CNN, mean pool | Vidal | 0.546 | 0.656 | 0.858 | 0.235 | 0.700 | 0.771 |
| CNN, attention pool | Baco | 0.483 | 0.480 | 0.452 | 0.513 | 0.526 | 0.486 |
| CNN, attention pool | Chardonnay | 0.565 | 0.672 | 0.205 | 0.926 | 0.600 | 0.305 |
| CNN, attention pool | Vidal | 0.562 | 0.664 | 0.852 | 0.272 | 0.709 | 0.774 |
