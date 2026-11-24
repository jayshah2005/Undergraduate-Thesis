# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| small ResNet + ECA, mean pool | 0.607 [0.580, 0.633] | 0.604 [0.578, 0.631] | 0.563 [0.525, 0.601] | 0.651 [0.614, 0.688] | 0.645 [0.616, 0.673] | 0.601 [0.571, 0.631] |
| small ResNet + ECA, attention pool | 0.600 [0.574, 0.625] | 0.596 [0.571, 0.622] | 0.543 [0.505, 0.581] | 0.656 [0.619, 0.690] | 0.640 [0.611, 0.669] | 0.587 [0.557, 0.617] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| small ResNet + ECA, mean pool | 223 | 173 | 123 | 229 |
| small ResNet + ECA, attention pool | 215 | 181 | 121 | 231 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| small ResNet + ECA, mean pool | Baco | 0.472 | 0.468 | 0.422 | 0.522 | 0.514 | 0.463 |
| small ResNet + ECA, mean pool | Chardonnay | 0.577 | 0.680 | 0.227 | 0.926 | 0.625 | 0.333 |
| small ResNet + ECA, mean pool | Vidal | 0.552 | 0.664 | 0.844 | 0.260 | 0.719 | 0.777 |
| small ResNet + ECA, attention pool | Baco | 0.466 | 0.460 | 0.393 | 0.540 | 0.505 | 0.442 |
| small ResNet + ECA, attention pool | Chardonnay | 0.565 | 0.672 | 0.205 | 0.926 | 0.600 | 0.305 |
| small ResNet + ECA, attention pool | Vidal | 0.546 | 0.656 | 0.832 | 0.260 | 0.716 | 0.770 |
