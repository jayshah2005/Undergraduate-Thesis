# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

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

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| small ResNet (no max pool), mean pool | 0.607 [0.581, 0.633] | 0.604 [0.579, 0.631] | 0.568 [0.533, 0.604] | 0.645 [0.608, 0.679] | 0.643 [0.615, 0.671] | 0.603 [0.573, 0.632] |
| small ResNet (no max pool), attention pool | 0.615 [0.590, 0.639] | 0.614 [0.590, 0.638] | 0.598 [0.563, 0.634] | 0.631 [0.597, 0.665] | 0.646 [0.621, 0.670] | 0.621 [0.593, 0.649] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| small ResNet (no max pool), mean pool | 225 | 171 | 125 | 227 |
| small ResNet (no max pool), attention pool | 237 | 159 | 130 | 222 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| small ResNet (no max pool), mean pool | Baco | 0.461 | 0.456 | 0.400 | 0.522 | 0.500 | 0.444 |
| small ResNet (no max pool), mean pool | Chardonnay | 0.594 | 0.692 | 0.261 | 0.926 | 0.657 | 0.374 |
| small ResNet (no max pool), mean pool | Vidal | 0.545 | 0.664 | 0.855 | 0.234 | 0.715 | 0.779 |
| small ResNet (no max pool), attention pool | Baco | 0.471 | 0.468 | 0.437 | 0.504 | 0.513 | 0.472 |
| small ResNet (no max pool), attention pool | Chardonnay | 0.602 | 0.700 | 0.273 | 0.932 | 0.686 | 0.390 |
| small ResNet (no max pool), attention pool | Vidal | 0.536 | 0.672 | 0.890 | 0.182 | 0.710 | 0.790 |
