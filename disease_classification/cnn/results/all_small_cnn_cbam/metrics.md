# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| CNN + CBAM, mean pool | 0.610 [0.591, 0.628] | 0.579 [0.560, 0.596] | 0.706 [0.675, 0.737] | 0.513 [0.491, 0.533] | 0.429 [0.413, 0.443] | 0.533 [0.514, 0.552] |
| CNN + CBAM, attention pool | 0.624 [0.604, 0.643] | 0.583 [0.564, 0.602] | 0.753 [0.722, 0.784] | 0.495 [0.469, 0.519] | 0.435 [0.420, 0.451] | 0.552 [0.532, 0.571] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| CNN + CBAM, mean pool | 180 | 75 | 240 | 253 |
| CNN + CBAM, attention pool | 192 | 63 | 249 | 244 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| CNN + CBAM, mean pool | Baco | 0.497 | 0.460 | 0.766 | 0.227 | 0.429 | 0.550 |
| CNN + CBAM, mean pool | Chardonnay | 0.456 | 0.448 | 0.787 | 0.125 | 0.462 | 0.582 |
| CNN + CBAM, mean pool | Vidal | 0.496 | 0.828 | 0.077 | 0.915 | 0.095 | 0.085 |
| CNN + CBAM, attention pool | Baco | 0.497 | 0.460 | 0.766 | 0.227 | 0.429 | 0.550 |
| CNN + CBAM, attention pool | Chardonnay | 0.508 | 0.500 | 0.861 | 0.156 | 0.493 | 0.627 |
| CNN + CBAM, attention pool | Vidal | 0.525 | 0.788 | 0.192 | 0.857 | 0.135 | 0.159 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| CNN + CBAM, mean pool | 0.702 [0.683, 0.720] | 0.680 [0.662, 0.697] | 0.625 [0.603, 0.647] | 0.778 [0.748, 0.806] | 0.833 [0.814, 0.851] | 0.714 [0.696, 0.731] |
| CNN + CBAM, attention pool | 0.701 [0.680, 0.721] | 0.680 [0.660, 0.699] | 0.626 [0.601, 0.648] | 0.776 [0.745, 0.806] | 0.831 [0.811, 0.852] | 0.714 [0.694, 0.733] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| CNN + CBAM, mean pool | 398 | 239 | 80 | 281 |
| CNN + CBAM, attention pool | 399 | 238 | 81 | 280 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| CNN + CBAM, mean pool | Baco | 0.489 | 0.456 | 0.111 | 0.867 | 0.500 | 0.182 |
| CNN + CBAM, mean pool | Cab Franc | 0.522 | 0.904 | 0.934 | 0.111 | 0.966 | 0.949 |
| CNN + CBAM, mean pool | Chardonnay | 0.558 | 0.680 | 0.148 | 0.969 | 0.722 | 0.245 |
| CNN + CBAM, mean pool | Vidal | 0.581 | 0.680 | 0.838 | 0.325 | 0.736 | 0.784 |
| CNN + CBAM, attention pool | Baco | 0.503 | 0.472 | 0.148 | 0.858 | 0.556 | 0.234 |
| CNN + CBAM, attention pool | Cab Franc | 0.537 | 0.932 | 0.963 | 0.111 | 0.967 | 0.965 |
| CNN + CBAM, attention pool | Chardonnay | 0.584 | 0.696 | 0.205 | 0.963 | 0.750 | 0.321 |
| CNN + CBAM, attention pool | Vidal | 0.542 | 0.620 | 0.746 | 0.338 | 0.717 | 0.731 |
