# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| small ResNet (no max pool), mean pool | 0.557 [0.527, 0.589] | 0.560 [0.532, 0.590] | 0.549 [0.502, 0.596] | 0.566 [0.529, 0.600] | 0.395 [0.366, 0.426] | 0.460 [0.426, 0.495] |
| small ResNet (no max pool), attention pool | 0.552 [0.521, 0.581] | 0.543 [0.515, 0.571] | 0.580 [0.533, 0.627] | 0.523 [0.489, 0.558] | 0.386 [0.360, 0.412] | 0.464 [0.431, 0.494] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| small ResNet (no max pool), mean pool | 140 | 115 | 214 | 279 |
| small ResNet (no max pool), attention pool | 148 | 107 | 235 | 258 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| small ResNet (no max pool), mean pool | Baco | 0.477 | 0.480 | 0.458 | 0.496 | 0.408 | 0.432 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.431 | 0.424 | 0.713 | 0.148 | 0.444 | 0.547 |
| small ResNet (no max pool), mean pool | Vidal | 0.501 | 0.776 | 0.154 | 0.848 | 0.105 | 0.125 |
| small ResNet (no max pool), attention pool | Baco | 0.483 | 0.480 | 0.505 | 0.461 | 0.415 | 0.456 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.412 | 0.404 | 0.730 | 0.094 | 0.434 | 0.544 |
| small ResNet (no max pool), attention pool | Vidal | 0.500 | 0.744 | 0.192 | 0.808 | 0.104 | 0.135 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| small ResNet (no max pool), mean pool | 0.653 [0.634, 0.672] | 0.624 [0.605, 0.642] | 0.548 [0.523, 0.573] | 0.759 [0.731, 0.787] | 0.800 [0.781, 0.820] | 0.651 [0.629, 0.671] |
| small ResNet (no max pool), attention pool | 0.659 [0.641, 0.678] | 0.623 [0.604, 0.643] | 0.531 [0.504, 0.557] | 0.787 [0.762, 0.814] | 0.814 [0.795, 0.836] | 0.643 [0.620, 0.665] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| small ResNet (no max pool), mean pool | 349 | 288 | 87 | 274 |
| small ResNet (no max pool), attention pool | 338 | 299 | 77 | 284 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| small ResNet (no max pool), mean pool | Baco | 0.494 | 0.472 | 0.244 | 0.743 | 0.532 | 0.335 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.487 | 0.836 | 0.863 | 0.111 | 0.963 | 0.910 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.539 | 0.668 | 0.102 | 0.975 | 0.692 | 0.178 |
| small ResNet (no max pool), mean pool | Vidal | 0.487 | 0.520 | 0.572 | 0.403 | 0.683 | 0.623 |
| small ResNet (no max pool), attention pool | Baco | 0.509 | 0.484 | 0.230 | 0.788 | 0.564 | 0.326 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.479 | 0.820 | 0.846 | 0.111 | 0.962 | 0.901 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.536 | 0.664 | 0.102 | 0.969 | 0.643 | 0.176 |
| small ResNet (no max pool), attention pool | Vidal | 0.512 | 0.524 | 0.543 | 0.481 | 0.701 | 0.612 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | 0.546 [0.523, 0.571] | 0.517 [0.491, 0.544] | 0.649 [0.599, 0.694] | 0.837 [0.817, 0.857] | 0.639 [0.616, 0.663] |
| small ResNet (no max pool), mean pool | 0.587 [0.564, 0.611] | 0.501 [0.481, 0.521] | 0.432 [0.410, 0.454] | 0.743 [0.703, 0.784] | 0.855 [0.833, 0.875] | 0.574 [0.551, 0.596] |
| small ResNet (no max pool), attention pool | 0.591 [0.572, 0.613] | 0.507 [0.489, 0.525] | 0.439 [0.419, 0.460] | 0.743 [0.707, 0.779] | 0.857 [0.840, 0.875] | 0.581 [0.561, 0.600] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 401 | 375 | 78 | 144 |
| small ResNet (no max pool), mean pool | 335 | 441 | 57 | 165 |
| small ResNet (no max pool), attention pool | 341 | 435 | 57 | 165 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.440 | 0.387 | 0.339 | 0.542 | 0.703 | 0.457 |
| logistic regression | Cab Franc | 0.497 | 0.752 | 0.772 | 0.222 | 0.964 | 0.857 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.532 | 0.624 | 0.720 | 0.344 | 0.761 | 0.740 |
| small ResNet (no max pool), mean pool | Baco | 0.482 | 0.335 | 0.201 | 0.763 | 0.731 | 0.315 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.409 | 0.788 | 0.817 | 0.000 | 0.956 | 0.881 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.518 | 0.392 | 0.069 | 0.967 | 0.786 | 0.126 |
| small ResNet (no max pool), mean pool | Vidal | 0.497 | 0.488 | 0.478 | 0.516 | 0.742 | 0.582 |
| small ResNet (no max pool), attention pool | Baco | 0.476 | 0.335 | 0.206 | 0.746 | 0.722 | 0.321 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.464 | 0.792 | 0.817 | 0.111 | 0.961 | 0.883 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.518 | 0.392 | 0.069 | 0.967 | 0.786 | 0.126 |
| small ResNet (no max pool), attention pool | Vidal | 0.511 | 0.508 | 0.505 | 0.516 | 0.752 | 0.605 |
