# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.609 [0.583, 0.634] | 0.611 [0.586, 0.636] | 0.636 [0.598, 0.672] | 0.582 [0.545, 0.619] | 0.632 [0.607, 0.655] | 0.634 [0.605, 0.660] |
| CNN + ECA, mean pool | 0.609 [0.583, 0.634] | 0.607 [0.582, 0.632] | 0.578 [0.540, 0.616] | 0.639 [0.605, 0.673] | 0.643 [0.617, 0.670] | 0.609 [0.579, 0.638] |
| CNN + ECA, attention pool | 0.610 [0.586, 0.635] | 0.610 [0.584, 0.635] | 0.596 [0.561, 0.634] | 0.625 [0.594, 0.656] | 0.641 [0.617, 0.666] | 0.618 [0.590, 0.645] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 252 | 144 | 147 | 205 |
| CNN + ECA, mean pool | 229 | 167 | 127 | 225 |
| CNN + ECA, attention pool | 236 | 160 | 132 | 220 |

Per variety:

| model | variety | bal. acc. | accuracy | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|---|---|
| logistic regression | Baco | 0.441 | 0.448 | 0.511 | 0.372 | 0.493 | 0.502 |
| logistic regression | Chardonnay | 0.610 | 0.700 | 0.307 | 0.914 | 0.659 | 0.419 |
| logistic regression | Vidal | 0.548 | 0.684 | 0.902 | 0.195 | 0.716 | 0.798 |
| CNN + ECA, mean pool | Baco | 0.463 | 0.460 | 0.422 | 0.504 | 0.504 | 0.460 |
| CNN + ECA, mean pool | Chardonnay | 0.591 | 0.692 | 0.250 | 0.932 | 0.667 | 0.364 |
| CNN + ECA, mean pool | Vidal | 0.544 | 0.668 | 0.867 | 0.221 | 0.714 | 0.783 |
| CNN + ECA, attention pool | Baco | 0.478 | 0.480 | 0.504 | 0.451 | 0.523 | 0.513 |
| CNN + ECA, attention pool | Chardonnay | 0.588 | 0.692 | 0.239 | 0.938 | 0.677 | 0.353 |
| CNN + ECA, attention pool | Vidal | 0.535 | 0.656 | 0.850 | 0.221 | 0.710 | 0.774 |
