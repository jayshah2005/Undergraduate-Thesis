# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN + CBAM, mean pool | 0.590 [0.561, 0.620] | 0.572 [0.543, 0.600] | 0.647 [0.600, 0.694] | 0.533 [0.501, 0.566] | 0.418 [0.392, 0.443] | 0.508 [0.476, 0.539] |
| CNN + CBAM, attention pool | 0.602 [0.575, 0.630] | 0.578 [0.552, 0.604] | 0.678 [0.635, 0.722] | 0.525 [0.493, 0.558] | 0.425 [0.403, 0.448] | 0.523 [0.495, 0.551] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| CNN + CBAM, mean pool | 165 | 90 | 230 | 263 |
| CNN + CBAM, attention pool | 173 | 82 | 234 | 259 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| CNN + CBAM, mean pool | Baco | 0.534 | 0.512 | 0.692 | 0.376 | 0.457 | 0.550 |
| CNN + CBAM, mean pool | Chardonnay | 0.446 | 0.440 | 0.713 | 0.180 | 0.453 | 0.554 |
| CNN + CBAM, mean pool | Vidal | 0.494 | 0.764 | 0.154 | 0.835 | 0.098 | 0.119 |
| CNN + CBAM, attention pool | Baco | 0.505 | 0.488 | 0.626 | 0.383 | 0.435 | 0.513 |
| CNN + CBAM, attention pool | Chardonnay | 0.504 | 0.496 | 0.828 | 0.180 | 0.490 | 0.616 |
| CNN + CBAM, attention pool | Vidal | 0.502 | 0.748 | 0.192 | 0.812 | 0.106 | 0.137 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| CNN + CBAM, mean pool | 0.678 [0.658, 0.698] | 0.644 [0.624, 0.664] | 0.556 [0.529, 0.581] | 0.801 [0.773, 0.828] | 0.831 [0.810, 0.852] | 0.666 [0.644, 0.687] |
| CNN + CBAM, attention pool | 0.672 [0.651, 0.693] | 0.637 [0.617, 0.658] | 0.546 [0.520, 0.571] | 0.798 [0.767, 0.831] | 0.827 [0.804, 0.851] | 0.658 [0.635, 0.681] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| CNN + CBAM, mean pool | 354 | 283 | 72 | 289 |
| CNN + CBAM, attention pool | 348 | 289 | 73 | 288 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| CNN + CBAM, mean pool | Baco | 0.501 | 0.480 | 0.259 | 0.743 | 0.547 | 0.352 |
| CNN + CBAM, mean pool | Cab Franc | 0.529 | 0.916 | 0.946 | 0.111 | 0.966 | 0.956 |
| CNN + CBAM, mean pool | Chardonnay | 0.537 | 0.672 | 0.080 | 0.994 | 0.875 | 0.146 |
| CNN + CBAM, mean pool | Vidal | 0.522 | 0.508 | 0.486 | 0.558 | 0.712 | 0.577 |
| CNN + CBAM, attention pool | Baco | 0.477 | 0.452 | 0.193 | 0.761 | 0.491 | 0.277 |
| CNN + CBAM, attention pool | Cab Franc | 0.524 | 0.908 | 0.938 | 0.111 | 0.966 | 0.952 |
| CNN + CBAM, attention pool | Chardonnay | 0.533 | 0.664 | 0.091 | 0.975 | 0.667 | 0.160 |
| CNN + CBAM, attention pool | Vidal | 0.534 | 0.524 | 0.509 | 0.558 | 0.721 | 0.597 |
