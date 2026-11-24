# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet + CBAM, mean pool | 0.580 [0.550, 0.610] | 0.570 [0.543, 0.598] | 0.612 [0.565, 0.663] | 0.548 [0.515, 0.580] | 0.412 [0.386, 0.438] | 0.492 [0.461, 0.525] |
| small ResNet + CBAM, attention pool | 0.579 [0.551, 0.606] | 0.580 [0.553, 0.607] | 0.576 [0.533, 0.624] | 0.582 [0.550, 0.613] | 0.416 [0.389, 0.443] | 0.484 [0.452, 0.514] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| small ResNet + CBAM, mean pool | 156 | 99 | 223 | 270 |
| small ResNet + CBAM, attention pool | 147 | 108 | 206 | 287 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| small ResNet + CBAM, mean pool | Baco | 0.498 | 0.488 | 0.570 | 0.426 | 0.430 | 0.490 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.439 | 0.432 | 0.746 | 0.133 | 0.450 | 0.562 |
| small ResNet + CBAM, mean pool | Vidal | 0.508 | 0.788 | 0.154 | 0.862 | 0.114 | 0.131 |
| small ResNet + CBAM, attention pool | Baco | 0.503 | 0.496 | 0.551 | 0.454 | 0.434 | 0.486 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.462 | 0.456 | 0.697 | 0.227 | 0.462 | 0.556 |
| small ResNet + CBAM, attention pool | Vidal | 0.491 | 0.788 | 0.115 | 0.866 | 0.091 | 0.102 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| small ResNet + CBAM, mean pool | 0.676 [0.654, 0.696] | 0.640 [0.620, 0.659] | 0.548 [0.521, 0.573] | 0.803 [0.770, 0.837] | 0.831 [0.807, 0.855] | 0.660 [0.637, 0.682] |
| small ResNet + CBAM, attention pool | 0.673 [0.653, 0.693] | 0.640 [0.620, 0.659] | 0.554 [0.527, 0.581] | 0.792 [0.762, 0.823] | 0.825 [0.803, 0.846] | 0.663 [0.640, 0.685] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| small ResNet + CBAM, mean pool | 349 | 288 | 71 | 290 |
| small ResNet + CBAM, attention pool | 353 | 284 | 75 | 286 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| small ResNet + CBAM, mean pool | Baco | 0.495 | 0.468 | 0.185 | 0.805 | 0.532 | 0.275 |
| small ResNet + CBAM, mean pool | Cab Franc | 0.508 | 0.876 | 0.905 | 0.111 | 0.965 | 0.934 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.524 | 0.656 | 0.080 | 0.969 | 0.583 | 0.140 |
| small ResNet + CBAM, mean pool | Vidal | 0.552 | 0.560 | 0.572 | 0.532 | 0.733 | 0.643 |
| small ResNet + CBAM, attention pool | Baco | 0.486 | 0.456 | 0.148 | 0.823 | 0.500 | 0.229 |
| small ResNet + CBAM, attention pool | Cab Franc | 0.504 | 0.868 | 0.896 | 0.111 | 0.964 | 0.929 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet + CBAM, attention pool | Vidal | 0.532 | 0.572 | 0.636 | 0.429 | 0.714 | 0.673 |
