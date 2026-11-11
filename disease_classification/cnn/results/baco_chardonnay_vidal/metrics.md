# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| CNN, mean pool | 0.604 [0.582, 0.626] | 0.594 [0.572, 0.615] | 0.638 [0.602, 0.673] | 0.571 [0.543, 0.599] | 0.433 [0.413, 0.454] | 0.516 [0.491, 0.541] |
| CNN, attention pool | 0.620 [0.598, 0.642] | 0.603 [0.583, 0.623] | 0.673 [0.634, 0.709] | 0.567 [0.545, 0.591] | 0.444 [0.425, 0.463] | 0.535 [0.512, 0.559] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| CNN, mean pool | 162 | 92 | 212 | 282 |
| CNN, attention pool | 171 | 83 | 214 | 280 |

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

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |
| CNN, mean pool | 0.608 [0.585, 0.631] | 0.612 [0.590, 0.635] | 0.689 [0.656, 0.719] | 0.528 [0.494, 0.559] | 0.616 [0.597, 0.636] | 0.651 [0.627, 0.673] |
| CNN, attention pool | 0.605 [0.583, 0.626] | 0.610 [0.587, 0.631] | 0.702 [0.668, 0.735] | 0.508 [0.480, 0.537] | 0.611 [0.593, 0.629] | 0.653 [0.631, 0.675] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |
| CNN, mean pool | 270 | 122 | 168 | 188 |
| CNN, attention pool | 275 | 117 | 175 | 181 |

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
