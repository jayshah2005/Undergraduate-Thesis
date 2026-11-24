# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN + ECA, mean pool | 0.566 [0.537, 0.594] | 0.561 [0.535, 0.588] | 0.580 [0.533, 0.624] | 0.552 [0.519, 0.584] | 0.401 [0.374, 0.427] | 0.474 [0.442, 0.505] |
| CNN + ECA, attention pool | 0.587 [0.558, 0.614] | 0.586 [0.559, 0.610] | 0.592 [0.545, 0.635] | 0.582 [0.552, 0.613] | 0.423 [0.396, 0.448] | 0.493 [0.461, 0.524] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| CNN + ECA, mean pool | 148 | 107 | 221 | 272 |
| CNN + ECA, attention pool | 151 | 104 | 206 | 287 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| CNN + ECA, mean pool | Baco | 0.479 | 0.480 | 0.477 | 0.482 | 0.411 | 0.442 |
| CNN + ECA, mean pool | Chardonnay | 0.436 | 0.428 | 0.770 | 0.102 | 0.450 | 0.568 |
| CNN + ECA, mean pool | Vidal | 0.484 | 0.776 | 0.115 | 0.853 | 0.083 | 0.097 |
| CNN + ECA, attention pool | Baco | 0.512 | 0.512 | 0.514 | 0.511 | 0.444 | 0.476 |
| CNN + ECA, attention pool | Chardonnay | 0.452 | 0.444 | 0.770 | 0.133 | 0.459 | 0.575 |
| CNN + ECA, attention pool | Vidal | 0.480 | 0.800 | 0.077 | 0.884 | 0.071 | 0.074 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| CNN + ECA, mean pool | 0.661 [0.641, 0.680] | 0.626 [0.607, 0.645] | 0.535 [0.510, 0.559] | 0.787 [0.759, 0.814] | 0.816 [0.794, 0.838] | 0.646 [0.625, 0.667] |
| CNN + ECA, attention pool | 0.672 [0.651, 0.691] | 0.636 [0.616, 0.656] | 0.543 [0.516, 0.568] | 0.801 [0.770, 0.831] | 0.828 [0.805, 0.851] | 0.656 [0.633, 0.677] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| CNN + ECA, mean pool | 341 | 296 | 77 | 284 |
| CNN + ECA, attention pool | 346 | 291 | 72 | 289 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| CNN + ECA, mean pool | Baco | 0.494 | 0.468 | 0.200 | 0.788 | 0.529 | 0.290 |
| CNN + ECA, mean pool | Cab Franc | 0.502 | 0.864 | 0.892 | 0.111 | 0.964 | 0.927 |
| CNN + ECA, mean pool | Chardonnay | 0.527 | 0.660 | 0.080 | 0.975 | 0.636 | 0.141 |
| CNN + ECA, mean pool | Vidal | 0.500 | 0.512 | 0.532 | 0.468 | 0.692 | 0.601 |
| CNN + ECA, attention pool | Baco | 0.481 | 0.452 | 0.148 | 0.814 | 0.488 | 0.227 |
| CNN + ECA, attention pool | Cab Franc | 0.508 | 0.876 | 0.905 | 0.111 | 0.965 | 0.934 |
| CNN + ECA, attention pool | Chardonnay | 0.542 | 0.672 | 0.102 | 0.981 | 0.750 | 0.180 |
| CNN + ECA, attention pool | Vidal | 0.526 | 0.544 | 0.572 | 0.481 | 0.712 | 0.635 |
