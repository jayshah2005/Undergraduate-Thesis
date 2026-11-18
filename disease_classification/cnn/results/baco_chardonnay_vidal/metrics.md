# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| CNN, mean pool | 0.619 [0.599, 0.639] | 0.620 [0.600, 0.640] | 0.644 [0.611, 0.674] | 0.594 [0.565, 0.616] | 0.641 [0.622, 0.658] | 0.642 [0.619, 0.665] |
| CNN, attention pool | 0.606 [0.584, 0.629] | 0.610 [0.587, 0.632] | 0.664 [0.631, 0.702] | 0.548 [0.520, 0.577] | 0.623 [0.604, 0.642] | 0.643 [0.620, 0.667] |
| small ResNet, mean pool | 0.595 [0.571, 0.619] | 0.595 [0.571, 0.619] | 0.588 [0.553, 0.624] | 0.602 [0.571, 0.634] | 0.625 [0.602, 0.648] | 0.606 [0.578, 0.633] |
| small ResNet, attention pool | 0.594 [0.571, 0.616] | 0.590 [0.566, 0.612] | 0.518 [0.482, 0.553] | 0.670 [0.639, 0.699] | 0.639 [0.612, 0.664] | 0.572 [0.542, 0.601] |
| small ResNet (no max pool), mean pool | 0.624 [0.599, 0.646] | 0.627 [0.603, 0.650] | 0.682 [0.646, 0.715] | 0.565 [0.531, 0.597] | 0.638 [0.616, 0.659] | 0.659 [0.635, 0.683] |
| small ResNet (no max pool), attention pool | 0.590 [0.566, 0.614] | 0.592 [0.568, 0.616] | 0.624 [0.591, 0.662] | 0.557 [0.526, 0.588] | 0.613 [0.591, 0.634] | 0.618 [0.593, 0.644] |
| ResNet-18, mean pool | 0.602 [0.580, 0.623] | 0.602 [0.579, 0.623] | 0.601 [0.568, 0.634] | 0.602 [0.574, 0.631] | 0.630 [0.608, 0.650] | 0.615 [0.590, 0.639] |
| ResNet-18, attention pool | 0.591 [0.567, 0.615] | 0.590 [0.565, 0.614] | 0.563 [0.528, 0.601] | 0.619 [0.585, 0.651] | 0.625 [0.599, 0.649] | 0.592 [0.565, 0.621] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| CNN, mean pool | 255 | 141 | 143 | 209 |
| CNN, attention pool | 263 | 133 | 159 | 193 |
| small ResNet, mean pool | 233 | 163 | 140 | 212 |
| small ResNet, attention pool | 205 | 191 | 116 | 236 |
| small ResNet (no max pool), mean pool | 270 | 126 | 153 | 199 |
| small ResNet (no max pool), attention pool | 247 | 149 | 156 | 196 |
| ResNet-18, mean pool | 238 | 158 | 140 | 212 |
| ResNet-18, attention pool | 223 | 173 | 134 | 218 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| CNN, mean pool | Baco | 0.497 | 0.508 | 0.622 | 0.372 | 0.542 | 0.579 |
| CNN, mean pool | Chardonnay | 0.557 | 0.672 | 0.170 | 0.944 | 0.625 | 0.268 |
| CNN, mean pool | Vidal | 0.542 | 0.680 | 0.902 | 0.182 | 0.712 | 0.796 |
| CNN, attention pool | Baco | 0.475 | 0.504 | 0.807 | 0.142 | 0.529 | 0.639 |
| CNN, attention pool | Chardonnay | 0.558 | 0.676 | 0.159 | 0.957 | 0.667 | 0.257 |
| CNN, attention pool | Vidal | 0.547 | 0.648 | 0.809 | 0.286 | 0.718 | 0.761 |
| small ResNet, mean pool | Baco | 0.481 | 0.492 | 0.607 | 0.354 | 0.529 | 0.566 |
| small ResNet, mean pool | Chardonnay | 0.545 | 0.652 | 0.182 | 0.907 | 0.516 | 0.269 |
| small ResNet, mean pool | Vidal | 0.553 | 0.640 | 0.780 | 0.325 | 0.722 | 0.750 |
| small ResNet, attention pool | Baco | 0.487 | 0.488 | 0.496 | 0.478 | 0.532 | 0.513 |
| small ResNet, attention pool | Chardonnay | 0.541 | 0.664 | 0.125 | 0.957 | 0.611 | 0.208 |
| small ResNet, attention pool | Vidal | 0.542 | 0.616 | 0.734 | 0.351 | 0.718 | 0.726 |
| small ResNet (no max pool), mean pool | Baco | 0.530 | 0.548 | 0.733 | 0.327 | 0.566 | 0.639 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.565 | 0.668 | 0.216 | 0.914 | 0.576 | 0.314 |
| small ResNet (no max pool), mean pool | Vidal | 0.530 | 0.664 | 0.879 | 0.182 | 0.707 | 0.784 |
| small ResNet (no max pool), attention pool | Baco | 0.451 | 0.476 | 0.726 | 0.177 | 0.513 | 0.601 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.527 | 0.656 | 0.091 | 0.963 | 0.571 | 0.157 |
| small ResNet (no max pool), attention pool | Vidal | 0.537 | 0.644 | 0.815 | 0.260 | 0.712 | 0.760 |
| ResNet-18, mean pool | Baco | 0.501 | 0.516 | 0.667 | 0.336 | 0.545 | 0.600 |
| ResNet-18, mean pool | Chardonnay | 0.535 | 0.660 | 0.114 | 0.957 | 0.588 | 0.190 |
| ResNet-18, mean pool | Vidal | 0.522 | 0.628 | 0.798 | 0.247 | 0.704 | 0.748 |
| ResNet-18, attention pool | Baco | 0.489 | 0.496 | 0.563 | 0.416 | 0.535 | 0.549 |
| ResNet-18, attention pool | Chardonnay | 0.542 | 0.652 | 0.170 | 0.914 | 0.517 | 0.256 |
| ResNet-18, attention pool | Vidal | 0.531 | 0.620 | 0.763 | 0.299 | 0.710 | 0.735 |
