# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

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

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | 0.628 [0.606, 0.651] | 0.594 [0.564, 0.624] | 0.688 [0.655, 0.721] | 0.767 [0.746, 0.789] | 0.670 [0.646, 0.693] |
| small ResNet (no max pool), mean pool | 0.665 [0.647, 0.684] | 0.641 [0.623, 0.660] | 0.577 [0.553, 0.600] | 0.753 [0.726, 0.781] | 0.802 [0.784, 0.822] | 0.671 [0.651, 0.690] |
| small ResNet (no max pool), attention pool | 0.656 [0.638, 0.674] | 0.613 [0.595, 0.631] | 0.498 [0.472, 0.521] | 0.814 [0.786, 0.841] | 0.822 [0.799, 0.844] | 0.620 [0.597, 0.642] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 376 | 257 | 114 | 251 |
| small ResNet (no max pool), mean pool | 365 | 268 | 90 | 275 |
| small ResNet (no max pool), attention pool | 315 | 318 | 68 | 297 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.458 | 0.448 | 0.341 | 0.575 | 0.489 | 0.402 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.567 | 0.688 | 0.159 | 0.975 | 0.778 | 0.264 |
| logistic regression | Vidal | 0.542 | 0.620 | 0.763 | 0.321 | 0.701 | 0.731 |
| small ResNet (no max pool), mean pool | Baco | 0.508 | 0.488 | 0.281 | 0.735 | 0.559 | 0.374 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.497 | 0.856 | 0.884 | 0.111 | 0.964 | 0.922 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.556 | 0.680 | 0.136 | 0.975 | 0.750 | 0.231 |
| small ResNet (no max pool), mean pool | Vidal | 0.505 | 0.540 | 0.604 | 0.407 | 0.680 | 0.639 |
| small ResNet (no max pool), attention pool | Baco | 0.494 | 0.464 | 0.156 | 0.832 | 0.525 | 0.240 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.487 | 0.836 | 0.863 | 0.111 | 0.963 | 0.910 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.519 | 0.656 | 0.057 | 0.981 | 0.625 | 0.104 |
| small ResNet (no max pool), attention pool | Vidal | 0.505 | 0.496 | 0.479 | 0.531 | 0.681 | 0.562 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | 0.545 [0.521, 0.570] | 0.513 [0.486, 0.540] | 0.655 [0.611, 0.699] | 0.835 [0.816, 0.856] | 0.636 [0.611, 0.660] |
| small ResNet (no max pool), mean pool | 0.577 [0.554, 0.598] | 0.495 [0.475, 0.514] | 0.427 [0.405, 0.449] | 0.726 [0.686, 0.761] | 0.842 [0.821, 0.862] | 0.567 [0.545, 0.588] |
| small ResNet (no max pool), attention pool | 0.599 [0.577, 0.621] | 0.511 [0.492, 0.529] | 0.438 [0.417, 0.460] | 0.761 [0.726, 0.796] | 0.862 [0.842, 0.881] | 0.581 [0.560, 0.601] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 376 | 78 | 148 |
| small ResNet (no max pool), mean pool | 330 | 442 | 62 | 164 |
| small ResNet (no max pool), attention pool | 338 | 434 | 54 | 172 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.449 | 0.391 | 0.339 | 0.559 | 0.711 | 0.459 |
| logistic regression | Cab Franc | 0.495 | 0.748 | 0.768 | 0.222 | 0.964 | 0.855 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.541 | 0.620 | 0.714 | 0.368 | 0.751 | 0.732 |
| small ResNet (no max pool), mean pool | Baco | 0.484 | 0.347 | 0.222 | 0.746 | 0.737 | 0.341 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.407 | 0.784 | 0.813 | 0.000 | 0.956 | 0.879 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.487 | 0.368 | 0.062 | 0.911 | 0.556 | 0.112 |
| small ResNet (no max pool), mean pool | Vidal | 0.505 | 0.480 | 0.451 | 0.559 | 0.732 | 0.558 |
| small ResNet (no max pool), attention pool | Baco | 0.493 | 0.343 | 0.206 | 0.780 | 0.750 | 0.324 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.530 | 0.816 | 0.838 | 0.222 | 0.967 | 0.898 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.518 | 0.396 | 0.081 | 0.956 | 0.765 | 0.147 |
| small ResNet (no max pool), attention pool | Vidal | 0.510 | 0.488 | 0.462 | 0.559 | 0.737 | 0.568 |
