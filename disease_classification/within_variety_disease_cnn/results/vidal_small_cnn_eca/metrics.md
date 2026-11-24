# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| CNN + ECA, mean pool | 0.453 [0.388, 0.518] | 0.508 [0.468, 0.548] | 0.385 [0.269, 0.500] | 0.522 [0.482, 0.567] | 0.085 [0.059, 0.111] | 0.140 [0.097, 0.182] |
| CNN + ECA, attention pool | 0.517 [0.455, 0.579] | 0.592 [0.548, 0.632] | 0.423 [0.308, 0.538] | 0.612 [0.567, 0.652] | 0.112 [0.082, 0.141] | 0.177 [0.130, 0.224] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| CNN + ECA, mean pool | 10 | 16 | 107 | 117 |
| CNN + ECA, attention pool | 11 | 15 | 87 | 137 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| CNN + ECA, mean pool | 0.502 [0.456, 0.547] | 0.500 [0.456, 0.544] | 0.497 [0.445, 0.549] | 0.506 [0.429, 0.571] | 0.694 [0.653, 0.732] | 0.579 [0.533, 0.623] |
| CNN + ECA, attention pool | 0.510 [0.469, 0.552] | 0.492 [0.452, 0.532] | 0.462 [0.410, 0.509] | 0.558 [0.494, 0.623] | 0.702 [0.663, 0.741] | 0.557 [0.511, 0.601] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| CNN + ECA, mean pool | 86 | 87 | 38 | 39 |
| CNN + ECA, attention pool | 80 | 93 | 34 | 43 |
