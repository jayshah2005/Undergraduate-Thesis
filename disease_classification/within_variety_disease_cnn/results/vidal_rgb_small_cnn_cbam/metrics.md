# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| CNN + CBAM, mean pool | 0.445 [0.398, 0.480] | 0.524 [0.484, 0.564] | 0.346 [0.269, 0.385] | 0.545 [0.500, 0.589] | 0.081 [0.062, 0.095] | 0.131 [0.101, 0.153] |
| CNN + CBAM, attention pool | 0.550 [0.507, 0.604] | 0.712 [0.684, 0.736] | 0.346 [0.269, 0.462] | 0.754 [0.723, 0.781] | 0.141 [0.109, 0.179] | 0.200 [0.156, 0.255] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| CNN + CBAM, mean pool | 9 | 17 | 102 | 122 |
| CNN + CBAM, attention pool | 9 | 17 | 55 | 169 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| CNN + CBAM, mean pool | 0.521 [0.462, 0.581] | 0.536 [0.480, 0.588] | 0.561 [0.491, 0.624] | 0.481 [0.390, 0.571] | 0.708 [0.662, 0.756] | 0.626 [0.571, 0.677] |
| CNN + CBAM, attention pool | 0.468 [0.415, 0.521] | 0.448 [0.400, 0.496] | 0.416 [0.364, 0.468] | 0.519 [0.429, 0.610] | 0.661 [0.608, 0.712] | 0.511 [0.459, 0.560] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| CNN + CBAM, mean pool | 97 | 76 | 40 | 37 |
| CNN + CBAM, attention pool | 72 | 101 | 37 | 40 |
