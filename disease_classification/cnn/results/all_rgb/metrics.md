# Disease: whole-leaf CNN on all varieties, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.542 [0.511, 0.573] | 0.544 [0.515, 0.572] | 0.537 [0.490, 0.588] | 0.548 [0.513, 0.580] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN, mean pool | 0.560 [0.531, 0.589] | 0.547 [0.519, 0.575] | 0.600 [0.553, 0.647] | 0.519 [0.485, 0.554] | 0.392 [0.367, 0.418] | 0.474 [0.443, 0.505] |
| CNN, attention pool | 0.572 [0.544, 0.598] | 0.580 [0.555, 0.604] | 0.545 [0.498, 0.588] | 0.598 [0.568, 0.627] | 0.412 [0.384, 0.439] | 0.470 [0.438, 0.501] |
| small ResNet, mean pool | 0.595 [0.568, 0.622] | 0.607 [0.582, 0.632] | 0.557 [0.510, 0.600] | 0.633 [0.602, 0.665] | 0.440 [0.411, 0.468] | 0.491 [0.459, 0.523] |
| small ResNet, attention pool | 0.564 [0.535, 0.592] | 0.567 [0.539, 0.594] | 0.557 [0.510, 0.600] | 0.572 [0.540, 0.602] | 0.402 [0.374, 0.429] | 0.467 [0.433, 0.499] |
| small ResNet (no max pool), mean pool | 0.557 [0.527, 0.589] | 0.560 [0.532, 0.590] | 0.549 [0.502, 0.596] | 0.566 [0.529, 0.600] | 0.395 [0.366, 0.426] | 0.460 [0.426, 0.495] |
| small ResNet (no max pool), attention pool | 0.552 [0.521, 0.581] | 0.543 [0.515, 0.571] | 0.580 [0.533, 0.627] | 0.523 [0.489, 0.558] | 0.386 [0.360, 0.412] | 0.464 [0.431, 0.494] |
| ResNet-18, mean pool | 0.586 [0.559, 0.614] | 0.584 [0.557, 0.611] | 0.592 [0.549, 0.635] | 0.580 [0.548, 0.613] | 0.422 [0.396, 0.448] | 0.493 [0.462, 0.523] |
| ResNet-18, attention pool | 0.610 [0.582, 0.637] | 0.594 [0.567, 0.620] | 0.663 [0.620, 0.706] | 0.558 [0.525, 0.590] | 0.437 [0.412, 0.461] | 0.526 [0.497, 0.554] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 118 | 223 | 270 |
| CNN, mean pool | 153 | 102 | 237 | 256 |
| CNN, attention pool | 139 | 116 | 198 | 295 |
| small ResNet, mean pool | 142 | 113 | 181 | 312 |
| small ResNet, attention pool | 142 | 113 | 211 | 282 |
| small ResNet (no max pool), mean pool | 140 | 115 | 214 | 279 |
| small ResNet (no max pool), attention pool | 148 | 107 | 235 | 258 |
| ResNet-18, mean pool | 151 | 104 | 207 | 286 |
| ResNet-18, attention pool | 169 | 86 | 218 | 275 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.459 | 0.468 | 0.393 | 0.525 | 0.385 | 0.389 |
| logistic regression | Chardonnay | 0.428 | 0.420 | 0.754 | 0.102 | 0.444 | 0.559 |
| logistic regression | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| CNN, mean pool | Baco | 0.454 | 0.448 | 0.505 | 0.404 | 0.391 | 0.441 |
| CNN, mean pool | Chardonnay | 0.456 | 0.448 | 0.787 | 0.125 | 0.462 | 0.582 |
| CNN, mean pool | Vidal | 0.466 | 0.744 | 0.115 | 0.817 | 0.068 | 0.086 |
| CNN, attention pool | Baco | 0.522 | 0.528 | 0.477 | 0.567 | 0.455 | 0.466 |
| CNN, attention pool | Chardonnay | 0.419 | 0.412 | 0.697 | 0.141 | 0.436 | 0.536 |
| CNN, attention pool | Vidal | 0.497 | 0.800 | 0.115 | 0.879 | 0.100 | 0.107 |
| small ResNet, mean pool | Baco | 0.524 | 0.532 | 0.467 | 0.582 | 0.459 | 0.463 |
| small ResNet, mean pool | Chardonnay | 0.525 | 0.520 | 0.746 | 0.305 | 0.506 | 0.603 |
| small ResNet, mean pool | Vidal | 0.446 | 0.768 | 0.038 | 0.853 | 0.029 | 0.033 |
| small ResNet, attention pool | Baco | 0.477 | 0.484 | 0.430 | 0.525 | 0.407 | 0.418 |
| small ResNet, attention pool | Chardonnay | 0.440 | 0.432 | 0.762 | 0.117 | 0.451 | 0.567 |
| small ResNet, attention pool | Vidal | 0.488 | 0.784 | 0.115 | 0.862 | 0.088 | 0.100 |
| small ResNet (no max pool), mean pool | Baco | 0.477 | 0.480 | 0.458 | 0.496 | 0.408 | 0.432 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.431 | 0.424 | 0.713 | 0.148 | 0.444 | 0.547 |
| small ResNet (no max pool), mean pool | Vidal | 0.501 | 0.776 | 0.154 | 0.848 | 0.105 | 0.125 |
| small ResNet (no max pool), attention pool | Baco | 0.483 | 0.480 | 0.505 | 0.461 | 0.415 | 0.456 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.412 | 0.404 | 0.730 | 0.094 | 0.434 | 0.544 |
| small ResNet (no max pool), attention pool | Vidal | 0.500 | 0.744 | 0.192 | 0.808 | 0.104 | 0.135 |
| ResNet-18, mean pool | Baco | 0.523 | 0.516 | 0.570 | 0.475 | 0.452 | 0.504 |
| ResNet-18, mean pool | Chardonnay | 0.509 | 0.504 | 0.713 | 0.305 | 0.494 | 0.584 |
| ResNet-18, mean pool | Vidal | 0.459 | 0.732 | 0.115 | 0.804 | 0.064 | 0.082 |
| ResNet-18, attention pool | Baco | 0.549 | 0.536 | 0.645 | 0.454 | 0.473 | 0.545 |
| ResNet-18, attention pool | Chardonnay | 0.471 | 0.464 | 0.779 | 0.164 | 0.470 | 0.586 |
| ResNet-18, attention pool | Vidal | 0.520 | 0.780 | 0.192 | 0.848 | 0.128 | 0.154 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.644 [0.622, 0.667] | 0.632 [0.609, 0.656] | 0.601 [0.573, 0.631] | 0.687 [0.654, 0.723] | 0.772 [0.752, 0.794] | 0.676 [0.653, 0.701] |
| CNN, mean pool | 0.660 [0.640, 0.680] | 0.625 [0.605, 0.645] | 0.534 [0.509, 0.557] | 0.787 [0.756, 0.814] | 0.815 [0.793, 0.838] | 0.645 [0.623, 0.666] |
| CNN, attention pool | 0.666 [0.645, 0.686] | 0.628 [0.607, 0.648] | 0.529 [0.502, 0.554] | 0.803 [0.773, 0.834] | 0.826 [0.802, 0.849] | 0.645 [0.621, 0.667] |
| small ResNet, mean pool | 0.659 [0.641, 0.678] | 0.630 [0.612, 0.648] | 0.554 [0.529, 0.578] | 0.765 [0.737, 0.792] | 0.806 [0.786, 0.826] | 0.657 [0.636, 0.676] |
| small ResNet, attention pool | 0.668 [0.646, 0.688] | 0.646 [0.625, 0.666] | 0.590 [0.562, 0.617] | 0.745 [0.715, 0.776] | 0.803 [0.782, 0.824] | 0.681 [0.657, 0.702] |
| small ResNet (no max pool), mean pool | 0.653 [0.634, 0.672] | 0.624 [0.605, 0.642] | 0.548 [0.523, 0.573] | 0.759 [0.731, 0.787] | 0.800 [0.781, 0.820] | 0.651 [0.629, 0.671] |
| small ResNet (no max pool), attention pool | 0.659 [0.641, 0.678] | 0.623 [0.604, 0.643] | 0.531 [0.504, 0.557] | 0.787 [0.762, 0.814] | 0.814 [0.795, 0.836] | 0.643 [0.620, 0.665] |
| ResNet-18, mean pool | 0.665 [0.643, 0.685] | 0.639 [0.618, 0.659] | 0.573 [0.545, 0.598] | 0.756 [0.726, 0.787] | 0.806 [0.784, 0.827] | 0.670 [0.647, 0.691] |
| ResNet-18, attention pool | 0.675 [0.655, 0.694] | 0.647 [0.628, 0.666] | 0.575 [0.549, 0.598] | 0.776 [0.745, 0.806] | 0.819 [0.798, 0.839] | 0.675 [0.655, 0.695] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 383 | 254 | 113 | 248 |
| CNN, mean pool | 340 | 297 | 77 | 284 |
| CNN, attention pool | 337 | 300 | 71 | 290 |
| small ResNet, mean pool | 353 | 284 | 85 | 276 |
| small ResNet, attention pool | 376 | 261 | 92 | 269 |
| small ResNet (no max pool), mean pool | 349 | 288 | 87 | 274 |
| small ResNet (no max pool), attention pool | 338 | 299 | 77 | 284 |
| ResNet-18, mean pool | 365 | 272 | 88 | 273 |
| ResNet-18, attention pool | 366 | 271 | 81 | 280 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.461 | 0.452 | 0.356 | 0.566 | 0.495 | 0.414 |
| logistic regression | Cab Franc | 0.499 | 0.756 | 0.776 | 0.222 | 0.964 | 0.860 |
| logistic regression | Chardonnay | 0.573 | 0.692 | 0.170 | 0.975 | 0.789 | 0.280 |
| logistic regression | Vidal | 0.540 | 0.628 | 0.769 | 0.312 | 0.715 | 0.741 |
| CNN, mean pool | Baco | 0.500 | 0.476 | 0.222 | 0.779 | 0.545 | 0.316 |
| CNN, mean pool | Cab Franc | 0.493 | 0.848 | 0.876 | 0.111 | 0.963 | 0.917 |
| CNN, mean pool | Chardonnay | 0.525 | 0.660 | 0.068 | 0.981 | 0.667 | 0.124 |
| CNN, mean pool | Vidal | 0.503 | 0.516 | 0.538 | 0.468 | 0.694 | 0.606 |
| CNN, attention pool | Baco | 0.492 | 0.464 | 0.170 | 0.814 | 0.523 | 0.257 |
| CNN, attention pool | Cab Franc | 0.495 | 0.852 | 0.880 | 0.111 | 0.964 | 0.920 |
| CNN, attention pool | Chardonnay | 0.539 | 0.672 | 0.091 | 0.988 | 0.800 | 0.163 |
| CNN, attention pool | Vidal | 0.512 | 0.524 | 0.543 | 0.481 | 0.701 | 0.612 |
| small ResNet, mean pool | Baco | 0.497 | 0.476 | 0.259 | 0.735 | 0.538 | 0.350 |
| small ResNet, mean pool | Cab Franc | 0.502 | 0.864 | 0.892 | 0.111 | 0.964 | 0.927 |
| small ResNet, mean pool | Chardonnay | 0.531 | 0.664 | 0.080 | 0.981 | 0.700 | 0.143 |
| small ResNet, mean pool | Vidal | 0.492 | 0.516 | 0.555 | 0.429 | 0.686 | 0.613 |
| small ResNet, attention pool | Baco | 0.508 | 0.488 | 0.281 | 0.735 | 0.559 | 0.374 |
| small ResNet, attention pool | Cab Franc | 0.485 | 0.832 | 0.859 | 0.111 | 0.963 | 0.908 |
| small ResNet, attention pool | Chardonnay | 0.542 | 0.672 | 0.102 | 0.981 | 0.750 | 0.180 |
| small ResNet, attention pool | Vidal | 0.521 | 0.592 | 0.705 | 0.338 | 0.705 | 0.705 |
| small ResNet (no max pool), mean pool | Baco | 0.494 | 0.472 | 0.244 | 0.743 | 0.532 | 0.335 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.487 | 0.836 | 0.863 | 0.111 | 0.963 | 0.910 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.539 | 0.668 | 0.102 | 0.975 | 0.692 | 0.178 |
| small ResNet (no max pool), mean pool | Vidal | 0.487 | 0.520 | 0.572 | 0.403 | 0.683 | 0.623 |
| small ResNet (no max pool), attention pool | Baco | 0.509 | 0.484 | 0.230 | 0.788 | 0.564 | 0.326 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.479 | 0.820 | 0.846 | 0.111 | 0.962 | 0.901 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.536 | 0.664 | 0.102 | 0.969 | 0.643 | 0.176 |
| small ResNet (no max pool), attention pool | Vidal | 0.512 | 0.524 | 0.543 | 0.481 | 0.701 | 0.612 |
| ResNet-18, mean pool | Baco | 0.480 | 0.456 | 0.207 | 0.752 | 0.500 | 0.293 |
| ResNet-18, mean pool | Cab Franc | 0.516 | 0.892 | 0.921 | 0.111 | 0.965 | 0.943 |
| ResNet-18, mean pool | Chardonnay | 0.537 | 0.656 | 0.136 | 0.938 | 0.545 | 0.218 |
| ResNet-18, mean pool | Vidal | 0.525 | 0.552 | 0.595 | 0.455 | 0.710 | 0.648 |
| ResNet-18, attention pool | Baco | 0.520 | 0.496 | 0.252 | 0.788 | 0.586 | 0.352 |
| ResNet-18, attention pool | Cab Franc | 0.527 | 0.912 | 0.942 | 0.111 | 0.966 | 0.954 |
| ResNet-18, attention pool | Chardonnay | 0.564 | 0.680 | 0.170 | 0.957 | 0.682 | 0.273 |
| ResNet-18, attention pool | Vidal | 0.487 | 0.500 | 0.520 | 0.455 | 0.682 | 0.590 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.583 [0.555, 0.610] | 0.546 [0.523, 0.571] | 0.517 [0.491, 0.544] | 0.649 [0.599, 0.694] | 0.837 [0.817, 0.857] | 0.639 [0.616, 0.663] |
| CNN, mean pool | 0.591 [0.569, 0.614] | 0.497 [0.478, 0.516] | 0.421 [0.399, 0.443] | 0.761 [0.721, 0.802] | 0.861 [0.840, 0.882] | 0.566 [0.544, 0.587] |
| CNN, attention pool | 0.594 [0.573, 0.616] | 0.506 [0.489, 0.525] | 0.436 [0.415, 0.457] | 0.752 [0.716, 0.788] | 0.860 [0.841, 0.880] | 0.578 [0.559, 0.599] |
| small ResNet, mean pool | 0.597 [0.574, 0.620] | 0.521 [0.502, 0.541] | 0.460 [0.437, 0.483] | 0.734 [0.698, 0.770] | 0.858 [0.840, 0.878] | 0.599 [0.578, 0.620] |
| small ResNet, attention pool | 0.598 [0.577, 0.619] | 0.512 [0.494, 0.530] | 0.443 [0.423, 0.463] | 0.752 [0.721, 0.788] | 0.862 [0.845, 0.880] | 0.586 [0.566, 0.604] |
| small ResNet (no max pool), mean pool | 0.587 [0.564, 0.611] | 0.501 [0.481, 0.521] | 0.432 [0.410, 0.454] | 0.743 [0.703, 0.784] | 0.855 [0.833, 0.875] | 0.574 [0.551, 0.596] |
| small ResNet (no max pool), attention pool | 0.591 [0.572, 0.613] | 0.507 [0.489, 0.525] | 0.439 [0.419, 0.460] | 0.743 [0.707, 0.779] | 0.857 [0.840, 0.875] | 0.581 [0.561, 0.600] |
| ResNet-18, mean pool | 0.640 [0.617, 0.662] | 0.583 [0.563, 0.602] | 0.537 [0.514, 0.559] | 0.743 [0.703, 0.784] | 0.880 [0.863, 0.896] | 0.667 [0.647, 0.686] |
| ResNet-18, attention pool | 0.620 [0.595, 0.646] | 0.579 [0.557, 0.601] | 0.546 [0.519, 0.572] | 0.694 [0.653, 0.734] | 0.862 [0.844, 0.880] | 0.669 [0.646, 0.690] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 401 | 375 | 78 | 144 |
| CNN, mean pool | 327 | 449 | 53 | 169 |
| CNN, attention pool | 338 | 438 | 55 | 167 |
| small ResNet, mean pool | 357 | 419 | 59 | 163 |
| small ResNet, attention pool | 344 | 432 | 55 | 167 |
| small ResNet (no max pool), mean pool | 335 | 441 | 57 | 165 |
| small ResNet (no max pool), attention pool | 341 | 435 | 57 | 165 |
| ResNet-18, mean pool | 417 | 359 | 57 | 165 |
| ResNet-18, attention pool | 424 | 352 | 68 | 154 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.440 | 0.387 | 0.339 | 0.542 | 0.703 | 0.457 |
| logistic regression | Cab Franc | 0.497 | 0.752 | 0.772 | 0.222 | 0.964 | 0.857 |
| logistic regression | Chardonnay | 0.542 | 0.420 | 0.106 | 0.978 | 0.895 | 0.190 |
| logistic regression | Vidal | 0.532 | 0.624 | 0.720 | 0.344 | 0.761 | 0.740 |
| CNN, mean pool | Baco | 0.482 | 0.335 | 0.201 | 0.763 | 0.731 | 0.315 |
| CNN, mean pool | Cab Franc | 0.536 | 0.828 | 0.851 | 0.222 | 0.967 | 0.905 |
| CNN, mean pool | Chardonnay | 0.486 | 0.364 | 0.050 | 0.922 | 0.533 | 0.091 |
| CNN, mean pool | Vidal | 0.509 | 0.460 | 0.409 | 0.609 | 0.752 | 0.530 |
| CNN, attention pool | Baco | 0.497 | 0.331 | 0.180 | 0.814 | 0.756 | 0.291 |
| CNN, attention pool | Cab Franc | 0.485 | 0.832 | 0.859 | 0.111 | 0.963 | 0.908 |
| CNN, attention pool | Chardonnay | 0.499 | 0.384 | 0.087 | 0.911 | 0.636 | 0.154 |
| CNN, attention pool | Vidal | 0.504 | 0.476 | 0.446 | 0.562 | 0.748 | 0.559 |
| small ResNet, mean pool | Baco | 0.508 | 0.375 | 0.254 | 0.763 | 0.774 | 0.382 |
| small ResNet, mean pool | Cab Franc | 0.491 | 0.844 | 0.871 | 0.111 | 0.963 | 0.915 |
| small ResNet, mean pool | Chardonnay | 0.497 | 0.384 | 0.094 | 0.900 | 0.625 | 0.163 |
| small ResNet, mean pool | Vidal | 0.507 | 0.480 | 0.452 | 0.562 | 0.750 | 0.564 |
| small ResNet, attention pool | Baco | 0.492 | 0.351 | 0.222 | 0.763 | 0.750 | 0.343 |
| small ResNet, attention pool | Cab Franc | 0.470 | 0.804 | 0.830 | 0.111 | 0.962 | 0.891 |
| small ResNet, attention pool | Chardonnay | 0.505 | 0.376 | 0.044 | 0.967 | 0.700 | 0.082 |
| small ResNet, attention pool | Vidal | 0.521 | 0.516 | 0.511 | 0.531 | 0.760 | 0.611 |
| small ResNet (no max pool), mean pool | Baco | 0.482 | 0.335 | 0.201 | 0.763 | 0.731 | 0.315 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.409 | 0.788 | 0.817 | 0.000 | 0.956 | 0.881 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.518 | 0.392 | 0.069 | 0.967 | 0.786 | 0.126 |
| small ResNet (no max pool), mean pool | Vidal | 0.497 | 0.488 | 0.478 | 0.516 | 0.742 | 0.582 |
| small ResNet (no max pool), attention pool | Baco | 0.476 | 0.335 | 0.206 | 0.746 | 0.722 | 0.321 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.464 | 0.792 | 0.817 | 0.111 | 0.961 | 0.883 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.518 | 0.392 | 0.069 | 0.967 | 0.786 | 0.126 |
| small ResNet (no max pool), attention pool | Vidal | 0.511 | 0.508 | 0.505 | 0.516 | 0.752 | 0.605 |
| ResNet-18, mean pool | Baco | 0.600 | 0.532 | 0.471 | 0.729 | 0.848 | 0.605 |
| ResNet-18, mean pool | Cab Franc | 0.677 | 0.892 | 0.909 | 0.444 | 0.978 | 0.942 |
| ResNet-18, mean pool | Chardonnay | 0.556 | 0.460 | 0.212 | 0.900 | 0.791 | 0.335 |
| ResNet-18, mean pool | Vidal | 0.491 | 0.448 | 0.403 | 0.578 | 0.735 | 0.521 |
| ResNet-18, attention pool | Baco | 0.564 | 0.504 | 0.450 | 0.678 | 0.817 | 0.580 |
| ResNet-18, attention pool | Cab Franc | 0.592 | 0.832 | 0.851 | 0.333 | 0.972 | 0.907 |
| ResNet-18, attention pool | Chardonnay | 0.565 | 0.484 | 0.275 | 0.856 | 0.772 | 0.406 |
| ResNet-18, attention pool | Vidal | 0.508 | 0.496 | 0.484 | 0.531 | 0.750 | 0.588 |
