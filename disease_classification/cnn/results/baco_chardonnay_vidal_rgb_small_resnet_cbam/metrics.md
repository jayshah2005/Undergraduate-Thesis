# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| small ResNet + CBAM, mean pool | 0.618 [0.594, 0.642] | 0.616 [0.592, 0.640] | 0.586 [0.551, 0.621] | 0.651 [0.614, 0.685] | 0.654 [0.627, 0.680] | 0.618 [0.591, 0.644] |
| small ResNet + CBAM, attention pool | 0.589 [0.562, 0.617] | 0.587 [0.559, 0.615] | 0.545 [0.508, 0.583] | 0.634 [0.594, 0.673] | 0.626 [0.596, 0.656] | 0.583 [0.552, 0.614] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| small ResNet + CBAM, mean pool | 232 | 164 | 123 | 229 |
| small ResNet + CBAM, attention pool | 216 | 180 | 129 | 223 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| small ResNet + CBAM, mean pool | Baco | 0.495 | 0.492 | 0.459 | 0.531 | 0.539 | 0.496 |
| small ResNet + CBAM, mean pool | Chardonnay | 0.583 | 0.688 | 0.227 | 0.938 | 0.667 | 0.339 |
| small ResNet + CBAM, mean pool | Vidal | 0.544 | 0.668 | 0.867 | 0.221 | 0.714 | 0.783 |
| small ResNet + CBAM, attention pool | Baco | 0.454 | 0.452 | 0.430 | 0.478 | 0.496 | 0.460 |
| small ResNet + CBAM, attention pool | Chardonnay | 0.565 | 0.672 | 0.205 | 0.926 | 0.600 | 0.305 |
| small ResNet + CBAM, attention pool | Vidal | 0.528 | 0.636 | 0.809 | 0.247 | 0.707 | 0.755 |
