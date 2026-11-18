# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet (no max pool), mean pool | 0.555 [0.525, 0.585] | 0.556 [0.528, 0.586] | 0.551 [0.504, 0.598] | 0.559 [0.526, 0.591] | 0.391 [0.363, 0.420] | 0.458 [0.424, 0.491] |
| small ResNet (no max pool), attention pool | 0.574 [0.546, 0.601] | 0.575 [0.548, 0.599] | 0.571 [0.524, 0.614] | 0.577 [0.547, 0.607] | 0.410 [0.383, 0.435] | 0.477 [0.445, 0.507] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| small ResNet (no max pool), mean pool | 140 | 114 | 218 | 276 |
| small ResNet (no max pool), attention pool | 145 | 109 | 209 | 285 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.469 | 0.748 | 0.120 | 0.818 | 0.068 | 0.087 |
| small ResNet (no max pool), mean pool | Baco | 0.468 | 0.468 | 0.467 | 0.468 | 0.400 | 0.431 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.427 | 0.420 | 0.713 | 0.141 | 0.442 | 0.545 |
| small ResNet (no max pool), mean pool | Vidal | 0.487 | 0.780 | 0.120 | 0.853 | 0.083 | 0.098 |
| small ResNet (no max pool), attention pool | Baco | 0.499 | 0.500 | 0.495 | 0.504 | 0.431 | 0.461 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.423 | 0.416 | 0.730 | 0.117 | 0.441 | 0.549 |
| small ResNet (no max pool), attention pool | Vidal | 0.502 | 0.808 | 0.120 | 0.884 | 0.103 | 0.111 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |
| small ResNet (no max pool), mean pool | 0.605 [0.579, 0.630] | 0.604 [0.578, 0.630] | 0.587 [0.548, 0.622] | 0.624 [0.590, 0.657] | 0.632 [0.606, 0.658] | 0.608 [0.579, 0.637] |
| small ResNet (no max pool), attention pool | 0.608 [0.582, 0.633] | 0.607 [0.582, 0.632] | 0.587 [0.548, 0.622] | 0.629 [0.595, 0.663] | 0.635 [0.609, 0.661] | 0.610 [0.581, 0.638] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |
| small ResNet (no max pool), mean pool | 230 | 162 | 134 | 222 |
| small ResNet (no max pool), attention pool | 230 | 162 | 132 | 224 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |
| small ResNet (no max pool), mean pool | Baco | 0.477 | 0.476 | 0.467 | 0.487 | 0.521 | 0.492 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.576 | 0.676 | 0.239 | 0.914 | 0.600 | 0.341 |
| small ResNet (no max pool), mean pool | Vidal | 0.549 | 0.660 | 0.864 | 0.235 | 0.702 | 0.775 |
| small ResNet (no max pool), attention pool | Baco | 0.467 | 0.464 | 0.430 | 0.504 | 0.509 | 0.466 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.594 | 0.692 | 0.261 | 0.926 | 0.657 | 0.374 |
| small ResNet (no max pool), attention pool | Vidal | 0.546 | 0.664 | 0.882 | 0.210 | 0.700 | 0.780 |
