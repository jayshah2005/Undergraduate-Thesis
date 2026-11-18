# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | 0.596 [0.568, 0.624] | 0.494 [0.439, 0.549] | 0.649 [0.617, 0.682] | 0.421 [0.387, 0.457] | 0.455 [0.414, 0.495] |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | 0.549 [0.529, 0.570] | 0.698 [0.667, 0.729] | 0.473 [0.448, 0.497] | 0.406 [0.390, 0.422] | 0.514 [0.493, 0.533] |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | 0.572 [0.549, 0.595] | 0.737 [0.698, 0.776] | 0.487 [0.460, 0.513] | 0.426 [0.408, 0.445] | 0.540 [0.516, 0.564] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 126 | 129 | 173 | 320 |
| small ResNet, mean pool | 178 | 77 | 260 | 233 |
| small ResNet, attention pool | 188 | 67 | 253 | 240 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.471 | 0.468 | 0.495 | 0.447 | 0.405 | 0.445 |
| logistic regression | Chardonnay | 0.478 | 0.476 | 0.574 | 0.383 | 0.470 | 0.517 |
| logistic regression | Vidal | 0.522 | 0.844 | 0.115 | 0.929 | 0.158 | 0.133 |
| small ResNet, mean pool | Baco | 0.503 | 0.460 | 0.822 | 0.184 | 0.433 | 0.568 |
| small ResNet, mean pool | Chardonnay | 0.442 | 0.436 | 0.697 | 0.188 | 0.450 | 0.547 |
| small ResNet, mean pool | Vidal | 0.505 | 0.752 | 0.192 | 0.817 | 0.109 | 0.139 |
| small ResNet, attention pool | Baco | 0.518 | 0.488 | 0.738 | 0.298 | 0.444 | 0.554 |
| small ResNet, attention pool | Chardonnay | 0.520 | 0.512 | 0.844 | 0.195 | 0.500 | 0.628 |
| small ResNet, attention pool | Vidal | 0.502 | 0.716 | 0.231 | 0.772 | 0.105 | 0.145 |

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | 0.602 [0.575, 0.628] | 0.631 [0.596, 0.667] | 0.568 [0.531, 0.608] | 0.622 [0.597, 0.648] | 0.627 [0.599, 0.653] |
| small ResNet, mean pool | 0.595 [0.571, 0.619] | 0.595 [0.571, 0.619] | 0.588 [0.553, 0.624] | 0.602 [0.571, 0.634] | 0.625 [0.602, 0.648] | 0.606 [0.578, 0.633] |
| small ResNet, attention pool | 0.594 [0.571, 0.616] | 0.590 [0.566, 0.612] | 0.518 [0.482, 0.553] | 0.670 [0.639, 0.699] | 0.639 [0.612, 0.664] | 0.572 [0.542, 0.601] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 250 | 146 | 152 | 200 |
| small ResNet, mean pool | 233 | 163 | 140 | 212 |
| small ResNet, attention pool | 205 | 191 | 116 | 236 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.462 | 0.472 | 0.570 | 0.354 | 0.513 | 0.540 |
| logistic regression | Chardonnay | 0.542 | 0.648 | 0.182 | 0.901 | 0.500 | 0.267 |
| logistic regression | Vidal | 0.545 | 0.684 | 0.908 | 0.182 | 0.714 | 0.799 |
| small ResNet, mean pool | Baco | 0.481 | 0.492 | 0.607 | 0.354 | 0.529 | 0.566 |
| small ResNet, mean pool | Chardonnay | 0.545 | 0.652 | 0.182 | 0.907 | 0.516 | 0.269 |
| small ResNet, mean pool | Vidal | 0.553 | 0.640 | 0.780 | 0.325 | 0.722 | 0.750 |
| small ResNet, attention pool | Baco | 0.487 | 0.488 | 0.496 | 0.478 | 0.532 | 0.513 |
| small ResNet, attention pool | Chardonnay | 0.541 | 0.664 | 0.125 | 0.957 | 0.611 | 0.208 |
| small ResNet, attention pool | Vidal | 0.542 | 0.616 | 0.734 | 0.351 | 0.718 | 0.726 |
