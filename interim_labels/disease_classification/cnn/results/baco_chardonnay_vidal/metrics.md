# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| CNN, mean pool | 0.604 [0.582, 0.626] | 0.594 [0.572, 0.615] | 0.638 [0.602, 0.673] | 0.571 [0.543, 0.599] | 0.433 [0.413, 0.454] | 0.516 [0.491, 0.541] |
| CNN, attention pool | 0.620 [0.598, 0.642] | 0.603 [0.583, 0.623] | 0.673 [0.634, 0.709] | 0.567 [0.545, 0.591] | 0.444 [0.425, 0.463] | 0.535 [0.512, 0.559] |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | 0.567 [0.547, 0.588] | 0.665 [0.630, 0.701] | 0.516 [0.492, 0.540] | 0.414 [0.397, 0.433] | 0.511 [0.487, 0.535] |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | 0.584 [0.561, 0.606] | 0.740 [0.701, 0.780] | 0.504 [0.478, 0.532] | 0.434 [0.415, 0.453] | 0.547 [0.524, 0.571] |
| small ResNet (no max pool), mean pool | 0.598 [0.577, 0.618] | 0.586 [0.564, 0.606] | 0.638 [0.602, 0.673] | 0.559 [0.530, 0.585] | 0.426 [0.407, 0.445] | 0.511 [0.488, 0.533] |
| small ResNet (no max pool), attention pool | 0.606 [0.585, 0.627] | 0.576 [0.557, 0.596] | 0.701 [0.665, 0.736] | 0.512 [0.490, 0.534] | 0.425 [0.408, 0.442] | 0.529 [0.507, 0.550] |
| ResNet-18, mean pool | 0.601 [0.572, 0.627] | 0.582 [0.555, 0.607] | 0.661 [0.614, 0.705] | 0.540 [0.508, 0.571] | 0.425 [0.400, 0.448] | 0.518 [0.488, 0.545] |
| ResNet-18, attention pool | 0.577 [0.549, 0.603] | 0.556 [0.531, 0.582] | 0.642 [0.598, 0.685] | 0.512 [0.482, 0.543] | 0.403 [0.380, 0.425] | 0.495 [0.467, 0.521] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| CNN, mean pool | 162 | 92 | 212 | 282 |
| CNN, attention pool | 171 | 83 | 214 | 280 |
| small ResNet, mean pool | 169 | 85 | 239 | 255 |
| small ResNet, attention pool | 188 | 66 | 245 | 249 |
| small ResNet (no max pool), mean pool | 162 | 92 | 218 | 276 |
| small ResNet (no max pool), attention pool | 178 | 76 | 241 | 253 |
| ResNet-18, mean pool | 168 | 86 | 227 | 267 |
| ResNet-18, attention pool | 163 | 91 | 241 | 253 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| CNN, mean pool | Baco | 0.519 | 0.488 | 0.748 | 0.291 | 0.444 | 0.557 |
| CNN, mean pool | Chardonnay | 0.476 | 0.472 | 0.639 | 0.312 | 0.470 | 0.542 |
| CNN, mean pool | Vidal | 0.527 | 0.820 | 0.160 | 0.893 | 0.143 | 0.151 |
| CNN, attention pool | Baco | 0.525 | 0.496 | 0.738 | 0.312 | 0.449 | 0.558 |
| CNN, attention pool | Chardonnay | 0.521 | 0.516 | 0.713 | 0.328 | 0.503 | 0.590 |
| CNN, attention pool | Vidal | 0.531 | 0.796 | 0.200 | 0.862 | 0.139 | 0.164 |
| small ResNet, mean pool | Baco | 0.510 | 0.472 | 0.785 | 0.234 | 0.438 | 0.562 |
| small ResNet, mean pool | Chardonnay | 0.445 | 0.440 | 0.664 | 0.227 | 0.450 | 0.536 |
| small ResNet, mean pool | Vidal | 0.509 | 0.788 | 0.160 | 0.858 | 0.111 | 0.131 |
| small ResNet, attention pool | Baco | 0.507 | 0.476 | 0.738 | 0.277 | 0.436 | 0.549 |
| small ResNet, attention pool | Chardonnay | 0.489 | 0.480 | 0.861 | 0.117 | 0.482 | 0.618 |
| small ResNet, attention pool | Vidal | 0.513 | 0.796 | 0.160 | 0.867 | 0.118 | 0.136 |
| small ResNet (no max pool), mean pool | Baco | 0.528 | 0.504 | 0.701 | 0.355 | 0.452 | 0.549 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.458 | 0.452 | 0.689 | 0.227 | 0.459 | 0.551 |
| small ResNet (no max pool), mean pool | Vidal | 0.498 | 0.800 | 0.120 | 0.876 | 0.097 | 0.107 |
| small ResNet (no max pool), attention pool | Baco | 0.538 | 0.512 | 0.729 | 0.348 | 0.459 | 0.563 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.467 | 0.460 | 0.762 | 0.172 | 0.467 | 0.579 |
| small ResNet (no max pool), attention pool | Vidal | 0.544 | 0.756 | 0.280 | 0.809 | 0.140 | 0.187 |
| ResNet-18, mean pool | Baco | 0.507 | 0.488 | 0.645 | 0.369 | 0.437 | 0.521 |
| ResNet-18, mean pool | Chardonnay | 0.494 | 0.488 | 0.762 | 0.227 | 0.484 | 0.592 |
| ResNet-18, mean pool | Vidal | 0.533 | 0.768 | 0.240 | 0.827 | 0.133 | 0.171 |
| ResNet-18, attention pool | Baco | 0.522 | 0.504 | 0.654 | 0.390 | 0.449 | 0.532 |
| ResNet-18, attention pool | Chardonnay | 0.443 | 0.436 | 0.730 | 0.156 | 0.452 | 0.558 |
| ResNet-18, attention pool | Vidal | 0.476 | 0.728 | 0.160 | 0.791 | 0.078 | 0.105 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |
| CNN, mean pool | 0.608 [0.585, 0.631] | 0.612 [0.590, 0.635] | 0.689 [0.656, 0.719] | 0.528 [0.494, 0.559] | 0.616 [0.597, 0.636] | 0.651 [0.627, 0.673] |
| CNN, attention pool | 0.605 [0.583, 0.626] | 0.610 [0.587, 0.631] | 0.702 [0.668, 0.735] | 0.508 [0.480, 0.537] | 0.611 [0.593, 0.629] | 0.653 [0.631, 0.675] |
| small ResNet, mean pool | 0.624 [0.605, 0.644] | 0.628 [0.608, 0.648] | 0.707 [0.676, 0.737] | 0.542 [0.517, 0.567] | 0.630 [0.613, 0.645] | 0.666 [0.646, 0.687] |
| small ResNet, attention pool | 0.604 [0.581, 0.627] | 0.606 [0.582, 0.630] | 0.638 [0.602, 0.671] | 0.570 [0.539, 0.601] | 0.620 [0.599, 0.642] | 0.629 [0.603, 0.653] |
| small ResNet (no max pool), mean pool | 0.616 [0.592, 0.639] | 0.616 [0.592, 0.640] | 0.628 [0.594, 0.661] | 0.604 [0.570, 0.635] | 0.636 [0.613, 0.659] | 0.632 [0.606, 0.657] |
| small ResNet (no max pool), attention pool | 0.610 [0.589, 0.630] | 0.614 [0.594, 0.634] | 0.694 [0.663, 0.727] | 0.525 [0.497, 0.551] | 0.617 [0.600, 0.634] | 0.653 [0.632, 0.675] |
| ResNet-18, mean pool | 0.590 [0.567, 0.614] | 0.591 [0.567, 0.615] | 0.605 [0.569, 0.640] | 0.576 [0.545, 0.610] | 0.611 [0.589, 0.634] | 0.608 [0.580, 0.633] |
| ResNet-18, attention pool | 0.577 [0.552, 0.601] | 0.578 [0.553, 0.602] | 0.597 [0.564, 0.633] | 0.556 [0.520, 0.590] | 0.597 [0.574, 0.620] | 0.597 [0.571, 0.623] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |
| CNN, mean pool | 270 | 122 | 168 | 188 |
| CNN, attention pool | 275 | 117 | 175 | 181 |
| small ResNet, mean pool | 277 | 115 | 163 | 193 |
| small ResNet, attention pool | 250 | 142 | 153 | 203 |
| small ResNet (no max pool), mean pool | 246 | 146 | 141 | 215 |
| small ResNet (no max pool), attention pool | 272 | 120 | 169 | 187 |
| ResNet-18, mean pool | 237 | 155 | 151 | 205 |
| ResNet-18, attention pool | 234 | 158 | 158 | 198 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.468 | 0.548 | 0.372 | 0.510 | 0.529 |
| logistic regression | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| logistic regression | Vidal | 0.546 | 0.668 | 0.893 | 0.198 | 0.699 | 0.784 |
| CNN, mean pool | Baco | 0.483 | 0.512 | 0.815 | 0.150 | 0.534 | 0.645 |
| CNN, mean pool | Chardonnay | 0.565 | 0.668 | 0.216 | 0.914 | 0.576 | 0.314 |
| CNN, mean pool | Vidal | 0.559 | 0.656 | 0.834 | 0.284 | 0.709 | 0.766 |
| CNN, attention pool | Baco | 0.452 | 0.488 | 0.859 | 0.044 | 0.518 | 0.646 |
| CNN, attention pool | Chardonnay | 0.577 | 0.684 | 0.216 | 0.938 | 0.655 | 0.325 |
| CNN, attention pool | Vidal | 0.562 | 0.656 | 0.828 | 0.296 | 0.711 | 0.765 |
| small ResNet, mean pool | Baco | 0.499 | 0.524 | 0.785 | 0.212 | 0.544 | 0.642 |
| small ResNet, mean pool | Chardonnay | 0.563 | 0.672 | 0.193 | 0.932 | 0.607 | 0.293 |
| small ResNet, mean pool | Vidal | 0.567 | 0.688 | 0.911 | 0.222 | 0.710 | 0.798 |
| small ResNet, attention pool | Baco | 0.476 | 0.496 | 0.704 | 0.248 | 0.528 | 0.603 |
| small ResNet, attention pool | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| small ResNet, attention pool | Vidal | 0.559 | 0.656 | 0.834 | 0.284 | 0.709 | 0.766 |
| small ResNet (no max pool), mean pool | Baco | 0.538 | 0.548 | 0.659 | 0.416 | 0.574 | 0.614 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.551 | 0.664 | 0.170 | 0.932 | 0.577 | 0.263 |
| small ResNet (no max pool), mean pool | Vidal | 0.525 | 0.636 | 0.840 | 0.210 | 0.689 | 0.757 |
| small ResNet (no max pool), attention pool | Baco | 0.474 | 0.508 | 0.859 | 0.088 | 0.530 | 0.655 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.538 | 0.660 | 0.125 | 0.951 | 0.579 | 0.206 |
| small ResNet (no max pool), attention pool | Vidal | 0.571 | 0.672 | 0.858 | 0.284 | 0.714 | 0.780 |
| ResNet-18, mean pool | Baco | 0.493 | 0.508 | 0.667 | 0.319 | 0.539 | 0.596 |
| ResNet-18, mean pool | Chardonnay | 0.533 | 0.640 | 0.170 | 0.895 | 0.469 | 0.250 |
| ResNet-18, mean pool | Vidal | 0.539 | 0.624 | 0.781 | 0.296 | 0.698 | 0.737 |
| ResNet-18, attention pool | Baco | 0.469 | 0.472 | 0.496 | 0.442 | 0.515 | 0.506 |
| ResNet-18, attention pool | Chardonnay | 0.526 | 0.608 | 0.250 | 0.802 | 0.407 | 0.310 |
| ResNet-18, attention pool | Vidal | 0.540 | 0.652 | 0.858 | 0.222 | 0.697 | 0.769 |
