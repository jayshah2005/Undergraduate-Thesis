# Disease: whole-leaf CNN on all varieties, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| CNN, mean pool | 0.586 [0.566, 0.606] | 0.561 [0.541, 0.582] | 0.663 [0.631, 0.694] | 0.509 [0.483, 0.535] | 0.411 [0.394, 0.429] | 0.508 [0.487, 0.527] |
| CNN, attention pool | 0.597 [0.578, 0.618] | 0.547 [0.528, 0.567] | 0.757 [0.725, 0.788] | 0.438 [0.414, 0.462] | 0.411 [0.396, 0.426] | 0.532 [0.514, 0.551] |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | 0.549 [0.529, 0.570] | 0.698 [0.667, 0.729] | 0.473 [0.448, 0.497] | 0.406 [0.390, 0.422] | 0.514 [0.493, 0.533] |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | 0.572 [0.549, 0.595] | 0.737 [0.698, 0.776] | 0.487 [0.460, 0.513] | 0.426 [0.408, 0.445] | 0.540 [0.516, 0.564] |
| small ResNet (no max pool), mean pool | 0.595 [0.575, 0.613] | 0.580 [0.560, 0.599] | 0.643 [0.612, 0.671] | 0.548 [0.521, 0.572] | 0.424 [0.405, 0.441] | 0.511 [0.490, 0.530] |
| small ResNet (no max pool), attention pool | 0.602 [0.582, 0.621] | 0.563 [0.544, 0.580] | 0.725 [0.694, 0.757] | 0.479 [0.456, 0.499] | 0.419 [0.404, 0.433] | 0.531 [0.512, 0.549] |
| ResNet-18, mean pool | 0.607 [0.580, 0.633] | 0.584 [0.559, 0.610] | 0.678 [0.635, 0.722] | 0.535 [0.505, 0.566] | 0.430 [0.408, 0.453] | 0.527 [0.499, 0.554] |
| ResNet-18, attention pool | 0.573 [0.546, 0.602] | 0.557 [0.532, 0.583] | 0.624 [0.576, 0.671] | 0.523 [0.493, 0.552] | 0.404 [0.380, 0.428] | 0.490 [0.461, 0.519] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| CNN, mean pool | 169 | 86 | 242 | 251 |
| CNN, attention pool | 193 | 62 | 277 | 216 |
| small ResNet, mean pool | 178 | 77 | 260 | 233 |
| small ResNet, attention pool | 188 | 67 | 253 | 240 |
| small ResNet (no max pool), mean pool | 164 | 91 | 223 | 270 |
| small ResNet (no max pool), attention pool | 185 | 70 | 257 | 236 |
| ResNet-18, mean pool | 173 | 82 | 229 | 264 |
| ResNet-18, attention pool | 159 | 96 | 235 | 258 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| CNN, mean pool | Baco | 0.494 | 0.456 | 0.776 | 0.213 | 0.428 | 0.551 |
| CNN, mean pool | Chardonnay | 0.457 | 0.452 | 0.664 | 0.250 | 0.458 | 0.542 |
| CNN, mean pool | Vidal | 0.518 | 0.776 | 0.192 | 0.844 | 0.125 | 0.152 |
| CNN, attention pool | Baco | 0.516 | 0.480 | 0.776 | 0.255 | 0.441 | 0.563 |
| CNN, attention pool | Chardonnay | 0.454 | 0.444 | 0.869 | 0.039 | 0.463 | 0.604 |
| CNN, attention pool | Vidal | 0.468 | 0.716 | 0.154 | 0.781 | 0.075 | 0.101 |
| small ResNet, mean pool | Baco | 0.503 | 0.460 | 0.822 | 0.184 | 0.433 | 0.568 |
| small ResNet, mean pool | Chardonnay | 0.442 | 0.436 | 0.697 | 0.188 | 0.450 | 0.547 |
| small ResNet, mean pool | Vidal | 0.505 | 0.752 | 0.192 | 0.817 | 0.109 | 0.139 |
| small ResNet, attention pool | Baco | 0.518 | 0.488 | 0.738 | 0.298 | 0.444 | 0.554 |
| small ResNet, attention pool | Chardonnay | 0.520 | 0.512 | 0.844 | 0.195 | 0.500 | 0.628 |
| small ResNet, attention pool | Vidal | 0.502 | 0.716 | 0.231 | 0.772 | 0.105 | 0.145 |
| small ResNet (no max pool), mean pool | Baco | 0.480 | 0.448 | 0.720 | 0.241 | 0.418 | 0.529 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.497 | 0.492 | 0.705 | 0.289 | 0.486 | 0.575 |
| small ResNet (no max pool), mean pool | Vidal | 0.463 | 0.800 | 0.038 | 0.888 | 0.038 | 0.038 |
| small ResNet (no max pool), attention pool | Baco | 0.524 | 0.484 | 0.813 | 0.234 | 0.446 | 0.576 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.475 | 0.468 | 0.762 | 0.188 | 0.472 | 0.583 |
| small ResNet (no max pool), attention pool | Vidal | 0.496 | 0.736 | 0.192 | 0.799 | 0.100 | 0.132 |
| ResNet-18, mean pool | Baco | 0.548 | 0.524 | 0.720 | 0.376 | 0.467 | 0.566 |
| ResNet-18, mean pool | Chardonnay | 0.502 | 0.496 | 0.746 | 0.258 | 0.489 | 0.591 |
| ResNet-18, mean pool | Vidal | 0.493 | 0.732 | 0.192 | 0.795 | 0.098 | 0.130 |
| ResNet-18, attention pool | Baco | 0.507 | 0.492 | 0.617 | 0.397 | 0.437 | 0.512 |
| ResNet-18, attention pool | Chardonnay | 0.419 | 0.412 | 0.721 | 0.117 | 0.438 | 0.545 |
| ResNet-18, attention pool | Vidal | 0.514 | 0.768 | 0.192 | 0.835 | 0.119 | 0.147 |

## Red blotch (637 + / 361 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.647 [0.624, 0.668] | 0.645 [0.624, 0.665] | 0.642 [0.615, 0.667] | 0.651 [0.615, 0.684] | 0.764 [0.744, 0.784] | 0.698 [0.677, 0.717] |
| CNN, mean pool | 0.691 [0.672, 0.709] | 0.671 [0.652, 0.688] | 0.620 [0.597, 0.642] | 0.762 [0.734, 0.792] | 0.821 [0.802, 0.840] | 0.707 [0.687, 0.725] |
| CNN, attention pool | 0.666 [0.646, 0.685] | 0.647 [0.627, 0.666] | 0.598 [0.575, 0.623] | 0.734 [0.704, 0.765] | 0.799 [0.779, 0.818] | 0.684 [0.664, 0.702] |
| small ResNet, mean pool | 0.682 [0.662, 0.702] | 0.660 [0.641, 0.679] | 0.603 [0.578, 0.626] | 0.762 [0.731, 0.792] | 0.817 [0.797, 0.837] | 0.694 [0.673, 0.713] |
| small ResNet, attention pool | 0.686 [0.668, 0.704] | 0.664 [0.646, 0.681] | 0.608 [0.584, 0.630] | 0.765 [0.737, 0.792] | 0.820 [0.802, 0.838] | 0.698 [0.680, 0.715] |
| small ResNet (no max pool), mean pool | 0.683 [0.665, 0.700] | 0.672 [0.655, 0.689] | 0.645 [0.623, 0.666] | 0.720 [0.695, 0.748] | 0.803 [0.787, 0.820] | 0.715 [0.698, 0.731] |
| small ResNet (no max pool), attention pool | 0.673 [0.653, 0.692] | 0.654 [0.634, 0.673] | 0.604 [0.581, 0.628] | 0.742 [0.712, 0.770] | 0.805 [0.785, 0.825] | 0.691 [0.671, 0.709] |
| ResNet-18, mean pool | 0.672 [0.647, 0.693] | 0.649 [0.627, 0.669] | 0.590 [0.565, 0.615] | 0.753 [0.712, 0.789] | 0.809 [0.783, 0.832] | 0.682 [0.661, 0.703] |
| ResNet-18, attention pool | 0.667 [0.643, 0.689] | 0.642 [0.620, 0.662] | 0.578 [0.554, 0.603] | 0.756 [0.715, 0.792] | 0.807 [0.781, 0.832] | 0.673 [0.653, 0.694] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 409 | 228 | 126 | 235 |
| CNN, mean pool | 395 | 242 | 86 | 275 |
| CNN, attention pool | 381 | 256 | 96 | 265 |
| small ResNet, mean pool | 384 | 253 | 86 | 275 |
| small ResNet, attention pool | 387 | 250 | 85 | 276 |
| small ResNet (no max pool), mean pool | 411 | 226 | 101 | 260 |
| small ResNet (no max pool), attention pool | 385 | 252 | 93 | 268 |
| ResNet-18, mean pool | 376 | 261 | 89 | 272 |
| ResNet-18, attention pool | 368 | 269 | 88 | 273 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.456 | 0.440 | 0.267 | 0.646 | 0.474 | 0.341 |
| logistic regression | Cab Franc | 0.475 | 0.812 | 0.838 | 0.111 | 0.962 | 0.896 |
| logistic regression | Chardonnay | 0.540 | 0.656 | 0.148 | 0.932 | 0.542 | 0.232 |
| logistic regression | Vidal | 0.522 | 0.672 | 0.913 | 0.130 | 0.702 | 0.794 |
| CNN, mean pool | Baco | 0.499 | 0.468 | 0.148 | 0.850 | 0.541 | 0.233 |
| CNN, mean pool | Cab Franc | 0.522 | 0.904 | 0.934 | 0.111 | 0.966 | 0.949 |
| CNN, mean pool | Chardonnay | 0.555 | 0.676 | 0.148 | 0.963 | 0.684 | 0.243 |
| CNN, mean pool | Vidal | 0.539 | 0.636 | 0.792 | 0.286 | 0.714 | 0.751 |
| CNN, attention pool | Baco | 0.500 | 0.472 | 0.185 | 0.814 | 0.543 | 0.276 |
| CNN, attention pool | Cab Franc | 0.510 | 0.880 | 0.909 | 0.111 | 0.965 | 0.936 |
| CNN, attention pool | Chardonnay | 0.539 | 0.652 | 0.159 | 0.920 | 0.519 | 0.243 |
| CNN, attention pool | Vidal | 0.505 | 0.584 | 0.711 | 0.299 | 0.695 | 0.703 |
| small ResNet, mean pool | Baco | 0.504 | 0.476 | 0.185 | 0.823 | 0.556 | 0.278 |
| small ResNet, mean pool | Cab Franc | 0.512 | 0.884 | 0.913 | 0.111 | 0.965 | 0.938 |
| small ResNet, mean pool | Chardonnay | 0.555 | 0.672 | 0.159 | 0.951 | 0.636 | 0.255 |
| small ResNet, mean pool | Vidal | 0.537 | 0.608 | 0.723 | 0.351 | 0.714 | 0.718 |
| small ResNet, attention pool | Baco | 0.474 | 0.440 | 0.081 | 0.867 | 0.423 | 0.137 |
| small ResNet, attention pool | Cab Franc | 0.524 | 0.908 | 0.938 | 0.111 | 0.966 | 0.952 |
| small ResNet, attention pool | Chardonnay | 0.538 | 0.664 | 0.114 | 0.963 | 0.625 | 0.192 |
| small ResNet, attention pool | Vidal | 0.541 | 0.644 | 0.809 | 0.273 | 0.714 | 0.759 |
| small ResNet (no max pool), mean pool | Baco | 0.494 | 0.460 | 0.111 | 0.876 | 0.517 | 0.183 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.531 | 0.920 | 0.950 | 0.111 | 0.966 | 0.958 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.554 | 0.664 | 0.182 | 0.926 | 0.571 | 0.276 |
| small ResNet (no max pool), mean pool | Vidal | 0.501 | 0.644 | 0.873 | 0.130 | 0.693 | 0.772 |
| small ResNet (no max pool), attention pool | Baco | 0.466 | 0.435 | 0.119 | 0.814 | 0.432 | 0.186 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.563 | 0.880 | 0.905 | 0.222 | 0.969 | 0.936 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.527 | 0.656 | 0.091 | 0.963 | 0.571 | 0.157 |
| small ResNet (no max pool), attention pool | Vidal | 0.530 | 0.644 | 0.827 | 0.234 | 0.708 | 0.763 |
| ResNet-18, mean pool | Baco | 0.460 | 0.431 | 0.133 | 0.788 | 0.429 | 0.203 |
| ResNet-18, mean pool | Cab Franc | 0.701 | 0.940 | 0.959 | 0.444 | 0.979 | 0.969 |
| ResNet-18, mean pool | Chardonnay | 0.545 | 0.652 | 0.182 | 0.907 | 0.516 | 0.269 |
| ResNet-18, mean pool | Vidal | 0.529 | 0.572 | 0.642 | 0.416 | 0.712 | 0.675 |
| ResNet-18, attention pool | Baco | 0.472 | 0.440 | 0.104 | 0.841 | 0.438 | 0.168 |
| ResNet-18, attention pool | Cab Franc | 0.537 | 0.932 | 0.963 | 0.111 | 0.967 | 0.965 |
| ResNet-18, attention pool | Chardonnay | 0.541 | 0.644 | 0.193 | 0.889 | 0.486 | 0.276 |
| ResNet-18, attention pool | Vidal | 0.518 | 0.552 | 0.607 | 0.429 | 0.705 | 0.652 |

## Any disease vs healthy (776 + / 222 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.556 [0.523, 0.588] | 0.587 [0.561, 0.613] | 0.612 [0.584, 0.640] | 0.500 [0.441, 0.554] | 0.811 [0.791, 0.830] | 0.698 [0.675, 0.719] |
| CNN, mean pool | 0.572 [0.550, 0.595] | 0.560 [0.542, 0.578] | 0.550 [0.531, 0.572] | 0.595 [0.554, 0.635] | 0.826 [0.811, 0.842] | 0.660 [0.645, 0.678] |
| CNN, attention pool | 0.604 [0.580, 0.628] | 0.591 [0.570, 0.613] | 0.581 [0.557, 0.606] | 0.626 [0.586, 0.671] | 0.845 [0.829, 0.861] | 0.689 [0.669, 0.708] |
| small ResNet, mean pool | 0.637 [0.613, 0.658] | 0.565 [0.545, 0.585] | 0.508 [0.485, 0.531] | 0.766 [0.725, 0.806] | 0.883 [0.865, 0.901] | 0.645 [0.624, 0.666] |
| small ResNet, attention pool | 0.627 [0.601, 0.650] | 0.609 [0.588, 0.629] | 0.595 [0.571, 0.619] | 0.658 [0.617, 0.698] | 0.859 [0.843, 0.874] | 0.703 [0.684, 0.722] |
| small ResNet (no max pool), mean pool | 0.607 [0.580, 0.635] | 0.577 [0.554, 0.600] | 0.553 [0.527, 0.580] | 0.662 [0.613, 0.707] | 0.851 [0.833, 0.870] | 0.670 [0.648, 0.692] |
| small ResNet (no max pool), attention pool | 0.592 [0.567, 0.615] | 0.530 [0.509, 0.552] | 0.481 [0.457, 0.505] | 0.703 [0.662, 0.743] | 0.850 [0.830, 0.868] | 0.614 [0.592, 0.637] |
| ResNet-18, mean pool | 0.602 [0.575, 0.630] | 0.584 [0.560, 0.608] | 0.570 [0.543, 0.598] | 0.635 [0.586, 0.685] | 0.845 [0.827, 0.863] | 0.681 [0.657, 0.703] |
| ResNet-18, attention pool | 0.621 [0.594, 0.646] | 0.588 [0.567, 0.610] | 0.562 [0.537, 0.588] | 0.680 [0.631, 0.725] | 0.860 [0.841, 0.878] | 0.680 [0.658, 0.700] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 475 | 301 | 111 | 111 |
| CNN, mean pool | 427 | 349 | 90 | 132 |
| CNN, attention pool | 451 | 325 | 83 | 139 |
| small ResNet, mean pool | 394 | 382 | 52 | 170 |
| small ResNet, attention pool | 462 | 314 | 76 | 146 |
| small ResNet (no max pool), mean pool | 429 | 347 | 75 | 147 |
| small ResNet (no max pool), attention pool | 373 | 403 | 66 | 156 |
| ResNet-18, mean pool | 442 | 334 | 81 | 141 |
| ResNet-18, attention pool | 436 | 340 | 71 | 151 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.448 | 0.496 | 0.540 | 0.356 | 0.729 | 0.620 |
| logistic regression | Cab Franc | 0.444 | 0.752 | 0.776 | 0.111 | 0.959 | 0.858 |
| logistic regression | Chardonnay | 0.481 | 0.420 | 0.263 | 0.700 | 0.609 | 0.367 |
| logistic regression | Vidal | 0.590 | 0.680 | 0.774 | 0.406 | 0.791 | 0.783 |
| CNN, mean pool | Baco | 0.428 | 0.359 | 0.296 | 0.559 | 0.683 | 0.413 |
| CNN, mean pool | Cab Franc | 0.500 | 0.860 | 0.888 | 0.111 | 0.964 | 0.924 |
| CNN, mean pool | Chardonnay | 0.526 | 0.456 | 0.275 | 0.778 | 0.688 | 0.393 |
| CNN, mean pool | Vidal | 0.523 | 0.564 | 0.608 | 0.438 | 0.758 | 0.675 |
| CNN, attention pool | Baco | 0.464 | 0.423 | 0.386 | 0.542 | 0.730 | 0.505 |
| CNN, attention pool | Cab Franc | 0.555 | 0.864 | 0.888 | 0.222 | 0.968 | 0.926 |
| CNN, attention pool | Chardonnay | 0.581 | 0.504 | 0.306 | 0.856 | 0.790 | 0.441 |
| CNN, attention pool | Vidal | 0.528 | 0.572 | 0.618 | 0.438 | 0.762 | 0.682 |
| small ResNet, mean pool | Baco | 0.513 | 0.347 | 0.196 | 0.831 | 0.787 | 0.314 |
| small ResNet, mean pool | Cab Franc | 0.557 | 0.868 | 0.892 | 0.222 | 0.968 | 0.929 |
| small ResNet, mean pool | Chardonnay | 0.563 | 0.456 | 0.181 | 0.944 | 0.853 | 0.299 |
| small ResNet, mean pool | Vidal | 0.569 | 0.588 | 0.608 | 0.531 | 0.790 | 0.687 |
| small ResNet, attention pool | Baco | 0.593 | 0.540 | 0.492 | 0.695 | 0.838 | 0.620 |
| small ResNet, attention pool | Cab Franc | 0.586 | 0.820 | 0.838 | 0.333 | 0.971 | 0.900 |
| small ResNet, attention pool | Chardonnay | 0.553 | 0.472 | 0.263 | 0.844 | 0.750 | 0.389 |
| small ResNet, attention pool | Vidal | 0.539 | 0.604 | 0.672 | 0.406 | 0.767 | 0.716 |
| small ResNet (no max pool), mean pool | Baco | 0.447 | 0.415 | 0.386 | 0.508 | 0.716 | 0.502 |
| small ResNet (no max pool), mean pool | Cab Franc | 0.588 | 0.824 | 0.842 | 0.333 | 0.971 | 0.902 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.553 | 0.468 | 0.250 | 0.856 | 0.755 | 0.376 |
| small ResNet (no max pool), mean pool | Vidal | 0.593 | 0.600 | 0.608 | 0.578 | 0.807 | 0.693 |
| small ResNet (no max pool), attention pool | Baco | 0.454 | 0.363 | 0.280 | 0.627 | 0.707 | 0.402 |
| small ResNet (no max pool), attention pool | Cab Franc | 0.563 | 0.776 | 0.793 | 0.333 | 0.970 | 0.872 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.556 | 0.444 | 0.156 | 0.956 | 0.862 | 0.265 |
| small ResNet (no max pool), attention pool | Vidal | 0.514 | 0.536 | 0.559 | 0.469 | 0.754 | 0.642 |
| ResNet-18, mean pool | Baco | 0.467 | 0.427 | 0.392 | 0.542 | 0.733 | 0.510 |
| ResNet-18, mean pool | Cab Franc | 0.555 | 0.864 | 0.888 | 0.222 | 0.968 | 0.926 |
| ResNet-18, mean pool | Chardonnay | 0.575 | 0.524 | 0.394 | 0.756 | 0.741 | 0.514 |
| ResNet-18, mean pool | Vidal | 0.549 | 0.520 | 0.489 | 0.609 | 0.784 | 0.603 |
| ResNet-18, attention pool | Baco | 0.554 | 0.516 | 0.481 | 0.627 | 0.805 | 0.603 |
| ResNet-18, attention pool | Cab Franc | 0.543 | 0.840 | 0.863 | 0.222 | 0.967 | 0.912 |
| ResNet-18, attention pool | Chardonnay | 0.561 | 0.500 | 0.344 | 0.778 | 0.733 | 0.468 |
| ResNet-18, attention pool | Vidal | 0.549 | 0.496 | 0.441 | 0.656 | 0.788 | 0.566 |
