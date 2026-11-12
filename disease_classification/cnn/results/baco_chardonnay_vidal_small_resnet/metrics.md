# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.545, 0.610] | 0.599 [0.572, 0.628] | 0.508 [0.453, 0.563] | 0.646 [0.615, 0.678] | 0.424 [0.391, 0.461] | 0.462 [0.423, 0.504] |
| small ResNet, mean pool | 0.591 [0.569, 0.614] | 0.567 [0.547, 0.588] | 0.665 [0.630, 0.701] | 0.516 [0.492, 0.540] | 0.414 [0.397, 0.433] | 0.511 [0.487, 0.535] |
| small ResNet, attention pool | 0.622 [0.598, 0.646] | 0.584 [0.561, 0.606] | 0.740 [0.701, 0.780] | 0.504 [0.478, 0.532] | 0.434 [0.415, 0.453] | 0.547 [0.524, 0.571] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 129 | 125 | 175 | 319 |
| small ResNet, mean pool | 169 | 85 | 239 | 255 |
| small ResNet, attention pool | 188 | 66 | 245 | 249 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.470 | 0.464 | 0.514 | 0.426 | 0.404 | 0.453 |
| logistic regression | Chardonnay | 0.482 | 0.480 | 0.582 | 0.383 | 0.473 | 0.522 |
| logistic regression | Vidal | 0.527 | 0.852 | 0.120 | 0.933 | 0.167 | 0.140 |
| small ResNet, mean pool | Baco | 0.510 | 0.472 | 0.785 | 0.234 | 0.438 | 0.562 |
| small ResNet, mean pool | Chardonnay | 0.445 | 0.440 | 0.664 | 0.227 | 0.450 | 0.536 |
| small ResNet, mean pool | Vidal | 0.509 | 0.788 | 0.160 | 0.858 | 0.111 | 0.131 |
| small ResNet, attention pool | Baco | 0.507 | 0.476 | 0.738 | 0.277 | 0.436 | 0.549 |
| small ResNet, attention pool | Chardonnay | 0.489 | 0.480 | 0.861 | 0.117 | 0.482 | 0.618 |
| small ResNet, attention pool | Vidal | 0.513 | 0.796 | 0.160 | 0.867 | 0.118 | 0.136 |

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.597 [0.570, 0.624] | 0.598 [0.571, 0.626] | 0.615 [0.579, 0.651] | 0.579 [0.539, 0.618] | 0.616 [0.591, 0.643] | 0.616 [0.587, 0.643] |
| small ResNet, mean pool | 0.624 [0.605, 0.644] | 0.628 [0.608, 0.648] | 0.707 [0.676, 0.737] | 0.542 [0.517, 0.567] | 0.630 [0.613, 0.645] | 0.666 [0.646, 0.687] |
| small ResNet, attention pool | 0.604 [0.581, 0.627] | 0.606 [0.582, 0.630] | 0.638 [0.602, 0.671] | 0.570 [0.539, 0.601] | 0.620 [0.599, 0.642] | 0.629 [0.603, 0.653] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 241 | 151 | 150 | 206 |
| small ResNet, mean pool | 277 | 115 | 163 | 193 |
| small ResNet, attention pool | 250 | 142 | 153 | 203 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.460 | 0.468 | 0.548 | 0.372 | 0.510 | 0.529 |
| logistic regression | Chardonnay | 0.548 | 0.656 | 0.182 | 0.914 | 0.533 | 0.271 |
| logistic regression | Vidal | 0.546 | 0.668 | 0.893 | 0.198 | 0.699 | 0.784 |
| small ResNet, mean pool | Baco | 0.499 | 0.524 | 0.785 | 0.212 | 0.544 | 0.642 |
| small ResNet, mean pool | Chardonnay | 0.563 | 0.672 | 0.193 | 0.932 | 0.607 | 0.293 |
| small ResNet, mean pool | Vidal | 0.567 | 0.688 | 0.911 | 0.222 | 0.710 | 0.798 |
| small ResNet, attention pool | Baco | 0.476 | 0.496 | 0.704 | 0.248 | 0.528 | 0.603 |
| small ResNet, attention pool | Chardonnay | 0.549 | 0.664 | 0.159 | 0.938 | 0.583 | 0.250 |
| small ResNet, attention pool | Vidal | 0.559 | 0.656 | 0.834 | 0.284 | 0.709 | 0.766 |
