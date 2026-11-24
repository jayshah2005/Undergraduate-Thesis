# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet + ECA, mean pool | 0.551 [0.522, 0.578] | 0.539 [0.512, 0.564] | 0.588 [0.541, 0.631] | 0.513 [0.483, 0.544] | 0.385 [0.360, 0.408] | 0.465 [0.434, 0.494] |
| small ResNet + ECA, attention pool | 0.612 [0.584, 0.637] | 0.619 [0.594, 0.643] | 0.588 [0.545, 0.631] | 0.635 [0.604, 0.663] | 0.455 [0.426, 0.481] | 0.513 [0.481, 0.542] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| small ResNet + ECA, mean pool | 150 | 105 | 240 | 253 |
| small ResNet + ECA, attention pool | 150 | 105 | 180 | 313 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| small ResNet + ECA, mean pool | Baco | 0.468 | 0.464 | 0.495 | 0.440 | 0.402 | 0.444 |
| small ResNet + ECA, mean pool | Chardonnay | 0.420 | 0.412 | 0.762 | 0.078 | 0.441 | 0.559 |
| small ResNet + ECA, mean pool | Vidal | 0.481 | 0.740 | 0.154 | 0.808 | 0.085 | 0.110 |
| small ResNet + ECA, attention pool | Baco | 0.538 | 0.544 | 0.495 | 0.582 | 0.473 | 0.484 |
| small ResNet + ECA, attention pool | Chardonnay | 0.499 | 0.492 | 0.779 | 0.219 | 0.487 | 0.599 |
| small ResNet + ECA, attention pool | Vidal | 0.492 | 0.820 | 0.077 | 0.906 | 0.087 | 0.082 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| small ResNet + ECA, mean pool | 0.679 [0.658, 0.699] | 0.653 [0.632, 0.673] | 0.586 [0.559, 0.612] | 0.773 [0.742, 0.801] | 0.820 [0.799, 0.840] | 0.683 [0.660, 0.704] |
| small ResNet + ECA, attention pool | 0.676 [0.655, 0.697] | 0.646 [0.625, 0.667] | 0.570 [0.542, 0.598] | 0.781 [0.753, 0.812] | 0.821 [0.800, 0.843] | 0.673 [0.649, 0.696] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| small ResNet + ECA, mean pool | 373 | 264 | 82 | 279 |
| small ResNet + ECA, attention pool | 363 | 274 | 79 | 282 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| small ResNet + ECA, mean pool | Baco | 0.510 | 0.488 | 0.259 | 0.761 | 0.565 | 0.355 |
| small ResNet + ECA, mean pool | Cab Franc | 0.497 | 0.856 | 0.884 | 0.111 | 0.964 | 0.922 |
| small ResNet + ECA, mean pool | Chardonnay | 0.534 | 0.668 | 0.080 | 0.988 | 0.778 | 0.144 |
| small ResNet + ECA, mean pool | Vidal | 0.549 | 0.600 | 0.682 | 0.416 | 0.724 | 0.702 |
| small ResNet + ECA, attention pool | Baco | 0.507 | 0.484 | 0.244 | 0.770 | 0.559 | 0.340 |
| small ResNet + ECA, attention pool | Cab Franc | 0.483 | 0.828 | 0.855 | 0.111 | 0.963 | 0.905 |
| small ResNet + ECA, attention pool | Chardonnay | 0.534 | 0.668 | 0.080 | 0.988 | 0.778 | 0.144 |
| small ResNet + ECA, attention pool | Vidal | 0.559 | 0.604 | 0.676 | 0.442 | 0.731 | 0.703 |
