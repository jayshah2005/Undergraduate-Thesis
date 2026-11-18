# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| CNN, mean pool | 0.611 [0.585, 0.637] | 0.607 [0.580, 0.632] | 0.543 [0.505, 0.583] | 0.679 [0.645, 0.713] | 0.655 [0.626, 0.685] | 0.594 [0.563, 0.624] |
| CNN, attention pool | 0.609 [0.583, 0.635] | 0.607 [0.580, 0.632] | 0.568 [0.530, 0.606] | 0.651 [0.616, 0.682] | 0.647 [0.619, 0.673] | 0.605 [0.575, 0.634] |
| small ResNet, mean pool | 0.604 [0.579, 0.630] | 0.602 [0.575, 0.627] | 0.556 [0.515, 0.593] | 0.653 [0.619, 0.688] | 0.643 [0.615, 0.671] | 0.596 [0.565, 0.625] |
| small ResNet, attention pool | 0.596 [0.568, 0.622] | 0.594 [0.566, 0.620] | 0.558 [0.518, 0.598] | 0.634 [0.597, 0.670] | 0.631 [0.603, 0.660] | 0.592 [0.561, 0.623] |
| small ResNet (no max pool), mean pool | 0.607 [0.581, 0.633] | 0.604 [0.579, 0.631] | 0.568 [0.533, 0.604] | 0.645 [0.608, 0.679] | 0.643 [0.615, 0.671] | 0.603 [0.573, 0.632] |
| small ResNet (no max pool), attention pool | 0.615 [0.590, 0.639] | 0.614 [0.590, 0.638] | 0.598 [0.563, 0.634] | 0.631 [0.597, 0.665] | 0.646 [0.621, 0.670] | 0.621 [0.593, 0.649] |
| ResNet-18, mean pool | 0.605 [0.578, 0.632] | 0.604 [0.578, 0.631] | 0.588 [0.553, 0.626] | 0.622 [0.585, 0.659] | 0.637 [0.609, 0.664] | 0.612 [0.584, 0.641] |
| ResNet-18, attention pool | 0.581 [0.557, 0.605] | 0.576 [0.552, 0.600] | 0.505 [0.472, 0.543] | 0.656 [0.625, 0.688] | 0.623 [0.596, 0.651] | 0.558 [0.529, 0.588] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| CNN, mean pool | 215 | 181 | 113 | 239 |
| CNN, attention pool | 225 | 171 | 123 | 229 |
| small ResNet, mean pool | 220 | 176 | 122 | 230 |
| small ResNet, attention pool | 221 | 175 | 129 | 223 |
| small ResNet (no max pool), mean pool | 225 | 171 | 125 | 227 |
| small ResNet (no max pool), attention pool | 237 | 159 | 130 | 222 |
| ResNet-18, mean pool | 233 | 163 | 133 | 219 |
| ResNet-18, attention pool | 200 | 196 | 121 | 231 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| CNN, mean pool | Baco | 0.493 | 0.484 | 0.385 | 0.602 | 0.536 | 0.448 |
| CNN, mean pool | Chardonnay | 0.580 | 0.684 | 0.227 | 0.932 | 0.645 | 0.336 |
| CNN, mean pool | Vidal | 0.543 | 0.652 | 0.827 | 0.260 | 0.715 | 0.767 |
| CNN, attention pool | Baco | 0.473 | 0.468 | 0.415 | 0.531 | 0.514 | 0.459 |
| CNN, attention pool | Chardonnay | 0.592 | 0.696 | 0.239 | 0.944 | 0.700 | 0.356 |
| CNN, attention pool | Vidal | 0.532 | 0.656 | 0.855 | 0.208 | 0.708 | 0.775 |
| small ResNet, mean pool | Baco | 0.477 | 0.472 | 0.415 | 0.540 | 0.519 | 0.461 |
| small ResNet, mean pool | Chardonnay | 0.568 | 0.676 | 0.205 | 0.932 | 0.621 | 0.308 |
| small ResNet, mean pool | Vidal | 0.539 | 0.656 | 0.844 | 0.234 | 0.712 | 0.772 |
| small ResNet, attention pool | Baco | 0.480 | 0.476 | 0.437 | 0.522 | 0.522 | 0.476 |
| small ResNet, attention pool | Chardonnay | 0.559 | 0.660 | 0.216 | 0.901 | 0.543 | 0.309 |
| small ResNet, attention pool | Vidal | 0.530 | 0.644 | 0.827 | 0.234 | 0.708 | 0.763 |
| small ResNet (no max pool), mean pool | Baco | 0.461 | 0.456 | 0.400 | 0.522 | 0.500 | 0.444 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.594 | 0.692 | 0.261 | 0.926 | 0.657 | 0.374 |
| small ResNet (no max pool), mean pool | Vidal | 0.545 | 0.664 | 0.855 | 0.234 | 0.715 | 0.779 |
| small ResNet (no max pool), attention pool | Baco | 0.471 | 0.468 | 0.437 | 0.504 | 0.513 | 0.472 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.602 | 0.700 | 0.273 | 0.932 | 0.686 | 0.390 |
| small ResNet (no max pool), attention pool | Vidal | 0.536 | 0.672 | 0.890 | 0.182 | 0.710 | 0.790 |
| ResNet-18, mean pool | Baco | 0.511 | 0.512 | 0.526 | 0.496 | 0.555 | 0.540 |
| ResNet-18, mean pool | Chardonnay | 0.546 | 0.644 | 0.216 | 0.877 | 0.487 | 0.299 |
| ResNet-18, mean pool | Vidal | 0.550 | 0.656 | 0.827 | 0.273 | 0.719 | 0.769 |
| ResNet-18, attention pool | Baco | 0.487 | 0.488 | 0.496 | 0.478 | 0.532 | 0.513 |
| ResNet-18, attention pool | Chardonnay | 0.528 | 0.644 | 0.136 | 0.920 | 0.480 | 0.212 |
| ResNet-18, attention pool | Vidal | 0.532 | 0.596 | 0.699 | 0.364 | 0.712 | 0.706 |
