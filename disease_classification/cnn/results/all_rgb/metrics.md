# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN, mean pool | 0.565 [0.535, 0.593] | 0.560 [0.533, 0.587] | 0.579 [0.531, 0.626] | 0.551 [0.518, 0.581] | 0.398 [0.372, 0.424] | 0.472 [0.439, 0.503] |
| CNN, attention pool | 0.580 [0.552, 0.607] | 0.591 [0.564, 0.616] | 0.547 [0.500, 0.591] | 0.613 [0.583, 0.642] | 0.421 [0.392, 0.449] | 0.476 [0.443, 0.508] |
| small ResNet, mean pool | 0.585 [0.556, 0.614] | 0.598 [0.570, 0.624] | 0.547 [0.500, 0.594] | 0.623 [0.591, 0.656] | 0.428 [0.398, 0.458] | 0.480 [0.445, 0.514] |
| small ResNet, attention pool | 0.564 [0.535, 0.590] | 0.563 [0.536, 0.588] | 0.567 [0.520, 0.610] | 0.561 [0.530, 0.591] | 0.399 [0.372, 0.424] | 0.468 [0.436, 0.498] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| CNN, mean pool | 147 | 107 | 222 | 272 |
| CNN, attention pool | 139 | 115 | 191 | 303 |
| small ResNet, mean pool | 139 | 115 | 186 | 308 |
| small ResNet, attention pool | 144 | 110 | 217 | 277 |

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
| small ResNet, mean pool | Baco | 0.495 | 0.500 | 0.458 | 0.532 | 0.426 | 0.441 |
| small ResNet, mean pool | Chardonnay | 0.474 | 0.468 | 0.721 | 0.227 | 0.471 | 0.570 |
| small ResNet, mean pool | Vidal | 0.493 | 0.824 | 0.080 | 0.907 | 0.087 | 0.083 |
| small ResNet, attention pool | Baco | 0.483 | 0.488 | 0.449 | 0.518 | 0.414 | 0.430 |
| small ResNet, attention pool | Chardonnay | 0.412 | 0.404 | 0.762 | 0.062 | 0.437 | 0.555 |
| small ResNet, attention pool | Vidal | 0.496 | 0.796 | 0.120 | 0.871 | 0.094 | 0.105 |

## Red blotch (633 + / 365 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.641 [0.619, 0.664] | 0.628 [0.606, 0.651] | 0.594 [0.564, 0.624] | 0.688 [0.655, 0.721] | 0.767 [0.746, 0.789] | 0.670 [0.646, 0.693] |
| CNN, mean pool | 0.650 [0.631, 0.668] | 0.610 [0.591, 0.628] | 0.502 [0.477, 0.526] | 0.797 [0.770, 0.825] | 0.811 [0.788, 0.832] | 0.620 [0.598, 0.642] |
| CNN, attention pool | 0.667 [0.647, 0.686] | 0.633 [0.614, 0.653] | 0.542 [0.517, 0.567] | 0.792 [0.764, 0.819] | 0.819 [0.798, 0.839] | 0.652 [0.630, 0.674] |
| small ResNet, mean pool | 0.662 [0.643, 0.680] | 0.622 [0.604, 0.641] | 0.515 [0.491, 0.539] | 0.808 [0.781, 0.836] | 0.823 [0.800, 0.846] | 0.634 [0.612, 0.656] |
| small ResNet, attention pool | 0.662 [0.643, 0.680] | 0.631 [0.612, 0.649] | 0.547 [0.523, 0.569] | 0.778 [0.751, 0.805] | 0.810 [0.789, 0.831] | 0.653 [0.632, 0.672] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 376 | 257 | 114 | 251 |
| CNN, mean pool | 318 | 315 | 74 | 291 |
| CNN, attention pool | 343 | 290 | 76 | 289 |
| small ResNet, mean pool | 326 | 307 | 70 | 295 |
| small ResNet, attention pool | 346 | 287 | 81 | 284 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.458 | 0.448 | 0.341 | 0.575 | 0.489 | 0.402 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.567 | 0.688 | 0.159 | 0.975 | 0.778 | 0.264 |
| logistic regression | Vidal | 0.542 | 0.620 | 0.763 | 0.321 | 0.701 | 0.731 |
| CNN, mean pool | Baco | 0.497 | 0.468 | 0.170 | 0.823 | 0.535 | 0.258 |
| CNN, mean pool | Cab Franc | 0.489 | 0.840 | 0.867 | 0.111 | 0.963 | 0.913 |
| CNN, mean pool | Chardonnay | 0.517 | 0.656 | 0.045 | 0.988 | 0.667 | 0.085 |
| CNN, mean pool | Vidal | 0.471 | 0.476 | 0.485 | 0.457 | 0.651 | 0.556 |
| CNN, attention pool | Baco | 0.503 | 0.476 | 0.193 | 0.814 | 0.553 | 0.286 |
| CNN, attention pool | Cab Franc | 0.502 | 0.864 | 0.892 | 0.111 | 0.964 | 0.927 |
| CNN, attention pool | Chardonnay | 0.534 | 0.668 | 0.080 | 0.988 | 0.778 | 0.144 |
| CNN, attention pool | Vidal | 0.503 | 0.524 | 0.562 | 0.444 | 0.679 | 0.615 |
| small ResNet, mean pool | Baco | 0.489 | 0.464 | 0.200 | 0.779 | 0.519 | 0.289 |
| small ResNet, mean pool | Cab Franc | 0.493 | 0.848 | 0.876 | 0.111 | 0.963 | 0.917 |
| small ResNet, mean pool | Chardonnay | 0.511 | 0.652 | 0.034 | 0.988 | 0.600 | 0.065 |
| small ResNet, mean pool | Vidal | 0.535 | 0.524 | 0.503 | 0.568 | 0.708 | 0.588 |
| small ResNet, attention pool | Baco | 0.512 | 0.488 | 0.244 | 0.779 | 0.569 | 0.342 |
| small ResNet, attention pool | Cab Franc | 0.491 | 0.844 | 0.871 | 0.111 | 0.963 | 0.915 |
| small ResNet, attention pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet, attention pool | Vidal | 0.506 | 0.528 | 0.568 | 0.444 | 0.681 | 0.619 |

## Any disease vs healthy (772 + / 226 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.584 [0.558, 0.612] | 0.545 [0.521, 0.570] | 0.513 [0.486, 0.540] | 0.655 [0.611, 0.699] | 0.835 [0.816, 0.856] | 0.636 [0.611, 0.660] |
| CNN, mean pool | 0.614 [0.592, 0.635] | 0.509 [0.492, 0.526] | 0.422 [0.403, 0.442] | 0.805 [0.770, 0.841] | 0.881 [0.860, 0.901] | 0.571 [0.552, 0.590] |
| CNN, attention pool | 0.597 [0.575, 0.619] | 0.509 [0.490, 0.529] | 0.437 [0.416, 0.460] | 0.757 [0.721, 0.796] | 0.860 [0.841, 0.880] | 0.579 [0.558, 0.601] |
| small ResNet, mean pool | 0.593 [0.572, 0.616] | 0.501 [0.482, 0.520] | 0.425 [0.402, 0.448] | 0.761 [0.721, 0.796] | 0.859 [0.839, 0.880] | 0.568 [0.546, 0.591] |
| small ResNet, attention pool | 0.597 [0.577, 0.619] | 0.503 [0.484, 0.523] | 0.425 [0.402, 0.449] | 0.770 [0.735, 0.805] | 0.863 [0.844, 0.883] | 0.569 [0.547, 0.592] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 396 | 376 | 78 | 148 |
| CNN, mean pool | 326 | 446 | 44 | 182 |
| CNN, attention pool | 337 | 435 | 55 | 171 |
| small ResNet, mean pool | 328 | 444 | 54 | 172 |
| small ResNet, attention pool | 328 | 444 | 52 | 174 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.449 | 0.391 | 0.339 | 0.559 | 0.711 | 0.459 |
| logistic regression | Cab Franc | 0.495 | 0.748 | 0.768 | 0.222 | 0.964 | 0.855 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.541 | 0.620 | 0.714 | 0.368 | 0.751 | 0.732 |
| CNN, mean pool | Baco | 0.487 | 0.306 | 0.143 | 0.831 | 0.730 | 0.239 |
| CNN, mean pool | Cab Franc | 0.563 | 0.880 | 0.905 | 0.222 | 0.969 | 0.936 |
| CNN, mean pool | Chardonnay | 0.500 | 0.376 | 0.056 | 0.944 | 0.643 | 0.103 |
| CNN, mean pool | Vidal | 0.536 | 0.472 | 0.396 | 0.676 | 0.766 | 0.522 |
| CNN, attention pool | Baco | 0.521 | 0.359 | 0.212 | 0.831 | 0.800 | 0.335 |
| CNN, attention pool | Cab Franc | 0.538 | 0.832 | 0.855 | 0.222 | 0.967 | 0.907 |
| CNN, attention pool | Chardonnay | 0.500 | 0.388 | 0.100 | 0.900 | 0.640 | 0.173 |
| CNN, attention pool | Vidal | 0.493 | 0.456 | 0.412 | 0.574 | 0.721 | 0.524 |
| small ResNet, mean pool | Baco | 0.497 | 0.331 | 0.180 | 0.814 | 0.756 | 0.291 |
| small ResNet, mean pool | Cab Franc | 0.541 | 0.836 | 0.859 | 0.222 | 0.967 | 0.910 |
| small ResNet, mean pool | Chardonnay | 0.497 | 0.384 | 0.094 | 0.900 | 0.625 | 0.163 |
| small ResNet, mean pool | Vidal | 0.499 | 0.452 | 0.396 | 0.603 | 0.727 | 0.512 |
| small ResNet, attention pool | Baco | 0.523 | 0.379 | 0.249 | 0.797 | 0.797 | 0.379 |
| small ResNet, attention pool | Cab Franc | 0.468 | 0.800 | 0.826 | 0.111 | 0.961 | 0.888 |
| small ResNet, attention pool | Chardonnay | 0.499 | 0.368 | 0.031 | 0.967 | 0.625 | 0.060 |
| small ResNet, attention pool | Vidal | 0.498 | 0.464 | 0.423 | 0.574 | 0.726 | 0.535 |
