# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | 0.545 [0.515, 0.574] | 0.539 [0.492, 0.591] | 0.549 [0.514, 0.581] | 0.381 [0.351, 0.410] | 0.446 [0.411, 0.480] |
| CNN, mean pool | 0.565 [0.535, 0.593] | 0.560 [0.533, 0.587] | 0.579 [0.531, 0.626] | 0.551 [0.518, 0.581] | 0.398 [0.372, 0.424] | 0.472 [0.439, 0.503] |
| CNN, attention pool | 0.580 [0.552, 0.607] | 0.591 [0.564, 0.616] | 0.547 [0.500, 0.591] | 0.613 [0.583, 0.642] | 0.421 [0.392, 0.449] | 0.476 [0.443, 0.508] |
| small ResNet, mean pool | 0.585 [0.556, 0.614] | 0.598 [0.570, 0.624] | 0.547 [0.500, 0.594] | 0.623 [0.591, 0.656] | 0.428 [0.398, 0.458] | 0.480 [0.445, 0.514] |
| small ResNet, attention pool | 0.564 [0.535, 0.590] | 0.563 [0.536, 0.588] | 0.567 [0.520, 0.610] | 0.561 [0.530, 0.591] | 0.399 [0.372, 0.424] | 0.468 [0.436, 0.498] |
| small ResNet (no max pool), mean pool | 0.555 [0.525, 0.585] | 0.556 [0.528, 0.586] | 0.551 [0.504, 0.598] | 0.559 [0.526, 0.591] | 0.391 [0.363, 0.420] | 0.458 [0.424, 0.491] |
| small ResNet (no max pool), attention pool | 0.574 [0.546, 0.601] | 0.575 [0.548, 0.599] | 0.571 [0.524, 0.614] | 0.577 [0.547, 0.607] | 0.410 [0.383, 0.435] | 0.477 [0.445, 0.507] |
| ResNet-18, mean pool | 0.577 [0.551, 0.600] | 0.551 [0.525, 0.575] | 0.657 [0.618, 0.697] | 0.496 [0.464, 0.526] | 0.401 [0.381, 0.421] | 0.499 [0.473, 0.522] |
| ResNet-18, attention pool | 0.596 [0.568, 0.623] | 0.575 [0.548, 0.600] | 0.661 [0.618, 0.705] | 0.530 [0.496, 0.563] | 0.420 [0.397, 0.443] | 0.514 [0.485, 0.542] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 137 | 117 | 223 | 271 |
| CNN, mean pool | 147 | 107 | 222 | 272 |
| CNN, attention pool | 139 | 115 | 191 | 303 |
| small ResNet, mean pool | 139 | 115 | 186 | 308 |
| small ResNet, attention pool | 144 | 110 | 217 | 277 |
| small ResNet (no max pool), mean pool | 140 | 114 | 218 | 276 |
| small ResNet (no max pool), attention pool | 145 | 109 | 209 | 285 |
| ResNet-18, mean pool | 167 | 87 | 249 | 245 |
| ResNet-18, attention pool | 168 | 86 | 232 | 262 |

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
| small ResNet (no max pool), mean pool | Baco | 0.468 | 0.468 | 0.467 | 0.468 | 0.400 | 0.431 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.427 | 0.420 | 0.713 | 0.141 | 0.442 | 0.545 |
| small ResNet (no max pool), mean pool | Vidal | 0.487 | 0.780 | 0.120 | 0.853 | 0.083 | 0.098 |
| small ResNet (no max pool), attention pool | Baco | 0.499 | 0.500 | 0.495 | 0.504 | 0.431 | 0.461 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.423 | 0.416 | 0.730 | 0.117 | 0.441 | 0.549 |
| small ResNet (no max pool), attention pool | Vidal | 0.502 | 0.808 | 0.120 | 0.884 | 0.103 | 0.111 |
| ResNet-18, mean pool | Baco | 0.498 | 0.476 | 0.664 | 0.333 | 0.430 | 0.522 |
| ResNet-18, mean pool | Chardonnay | 0.463 | 0.456 | 0.746 | 0.180 | 0.464 | 0.572 |
| ResNet-18, mean pool | Vidal | 0.489 | 0.720 | 0.200 | 0.778 | 0.091 | 0.125 |
| ResNet-18, attention pool | Baco | 0.530 | 0.512 | 0.664 | 0.397 | 0.455 | 0.540 |
| ResNet-18, attention pool | Chardonnay | 0.463 | 0.456 | 0.762 | 0.164 | 0.465 | 0.578 |
| ResNet-18, attention pool | Vidal | 0.491 | 0.756 | 0.160 | 0.822 | 0.091 | 0.116 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | 0.606 [0.580, 0.631] | 0.630 [0.592, 0.668] | 0.579 [0.542, 0.615] | 0.622 [0.598, 0.646] | 0.626 [0.599, 0.652] |
| CNN, mean pool | 0.602 [0.576, 0.628] | 0.600 [0.575, 0.627] | 0.569 [0.531, 0.607] | 0.635 [0.601, 0.669] | 0.632 [0.605, 0.659] | 0.599 [0.569, 0.628] |
| CNN, attention pool | 0.607 [0.582, 0.633] | 0.606 [0.580, 0.631] | 0.569 [0.531, 0.605] | 0.646 [0.612, 0.680] | 0.639 [0.611, 0.666] | 0.602 [0.573, 0.631] |
| small ResNet, mean pool | 0.608 [0.583, 0.633] | 0.606 [0.580, 0.631] | 0.564 [0.526, 0.602] | 0.652 [0.615, 0.685] | 0.641 [0.613, 0.668] | 0.600 [0.570, 0.629] |
| small ResNet, attention pool | 0.598 [0.572, 0.625] | 0.598 [0.572, 0.624] | 0.587 [0.551, 0.622] | 0.610 [0.573, 0.646] | 0.623 [0.597, 0.651] | 0.604 [0.576, 0.633] |
| small ResNet (no max pool), mean pool | 0.605 [0.579, 0.630] | 0.604 [0.578, 0.630] | 0.587 [0.548, 0.622] | 0.624 [0.590, 0.657] | 0.632 [0.606, 0.658] | 0.608 [0.579, 0.637] |
| small ResNet (no max pool), attention pool | 0.608 [0.582, 0.633] | 0.607 [0.582, 0.632] | 0.587 [0.548, 0.622] | 0.629 [0.595, 0.663] | 0.635 [0.609, 0.661] | 0.610 [0.581, 0.638] |
| ResNet-18, mean pool | 0.562 [0.536, 0.589] | 0.560 [0.533, 0.587] | 0.520 [0.485, 0.559] | 0.604 [0.567, 0.640] | 0.591 [0.563, 0.620] | 0.554 [0.523, 0.582] |
| ResNet-18, attention pool | 0.587 [0.562, 0.613] | 0.590 [0.564, 0.616] | 0.643 [0.607, 0.681] | 0.531 [0.494, 0.567] | 0.601 [0.579, 0.626] | 0.621 [0.595, 0.648] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 247 | 145 | 150 | 206 |
| CNN, mean pool | 223 | 169 | 130 | 226 |
| CNN, attention pool | 223 | 169 | 126 | 230 |
| small ResNet, mean pool | 221 | 171 | 124 | 232 |
| small ResNet, attention pool | 230 | 162 | 139 | 217 |
| small ResNet (no max pool), mean pool | 230 | 162 | 134 | 222 |
| small ResNet (no max pool), attention pool | 230 | 162 | 132 | 224 |
| ResNet-18, mean pool | 204 | 188 | 141 | 215 |
| ResNet-18, attention pool | 252 | 140 | 167 | 189 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.434 | 0.440 | 0.496 | 0.372 | 0.486 | 0.491 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.551 | 0.676 | 0.905 | 0.198 | 0.702 | 0.791 |
| CNN, mean pool | Baco | 0.455 | 0.452 | 0.415 | 0.496 | 0.496 | 0.452 |
| CNN, mean pool | Chardonnay | 0.591 | 0.692 | 0.250 | 0.932 | 0.667 | 0.364 |
| CNN, mean pool | Vidal | 0.546 | 0.656 | 0.858 | 0.235 | 0.700 | 0.771 |
| CNN, attention pool | Baco | 0.483 | 0.480 | 0.452 | 0.513 | 0.526 | 0.486 |
| CNN, attention pool | Chardonnay | 0.565 | 0.672 | 0.205 | 0.926 | 0.600 | 0.305 |
| CNN, attention pool | Vidal | 0.562 | 0.664 | 0.852 | 0.272 | 0.709 | 0.774 |
| small ResNet, mean pool | Baco | 0.473 | 0.468 | 0.415 | 0.531 | 0.514 | 0.459 |
| small ResNet, mean pool | Chardonnay | 0.583 | 0.688 | 0.227 | 0.938 | 0.667 | 0.339 |
| small ResNet, mean pool | Vidal | 0.552 | 0.660 | 0.858 | 0.247 | 0.704 | 0.773 |
| small ResNet, attention pool | Baco | 0.464 | 0.464 | 0.459 | 0.469 | 0.508 | 0.482 |
| small ResNet, attention pool | Chardonnay | 0.567 | 0.668 | 0.227 | 0.907 | 0.571 | 0.325 |
| small ResNet, attention pool | Vidal | 0.543 | 0.660 | 0.876 | 0.210 | 0.698 | 0.777 |
| small ResNet (no max pool), mean pool | Baco | 0.477 | 0.476 | 0.467 | 0.487 | 0.521 | 0.492 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.576 | 0.676 | 0.239 | 0.914 | 0.600 | 0.341 |
| small ResNet (no max pool), mean pool | Vidal | 0.549 | 0.660 | 0.864 | 0.235 | 0.702 | 0.775 |
| small ResNet (no max pool), attention pool | Baco | 0.467 | 0.464 | 0.430 | 0.504 | 0.509 | 0.466 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.594 | 0.692 | 0.261 | 0.926 | 0.657 | 0.374 |
| small ResNet (no max pool), attention pool | Vidal | 0.546 | 0.664 | 0.882 | 0.210 | 0.700 | 0.780 |
| ResNet-18, mean pool | Baco | 0.493 | 0.496 | 0.526 | 0.460 | 0.538 | 0.532 |
| ResNet-18, mean pool | Chardonnay | 0.494 | 0.596 | 0.148 | 0.840 | 0.333 | 0.205 |
| ResNet-18, mean pool | Vidal | 0.522 | 0.588 | 0.710 | 0.333 | 0.690 | 0.700 |
| ResNet-18, attention pool | Baco | 0.527 | 0.544 | 0.726 | 0.327 | 0.563 | 0.634 |
| ResNet-18, attention pool | Chardonnay | 0.514 | 0.596 | 0.239 | 0.790 | 0.382 | 0.294 |
| ResNet-18, attention pool | Vidal | 0.542 | 0.628 | 0.787 | 0.296 | 0.700 | 0.741 |
