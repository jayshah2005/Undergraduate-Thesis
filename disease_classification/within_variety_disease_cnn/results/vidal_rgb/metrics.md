# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | 0.460 [0.404, 0.516] | 0.360 [0.200, 0.520] | 0.471 [0.409, 0.529] | 0.070 [0.042, 0.098] | 0.118 [0.069, 0.164] |
| CNN, mean pool | 0.396 [0.327, 0.460] | 0.392 [0.344, 0.436] | 0.400 [0.280, 0.520] | 0.391 [0.342, 0.440] | 0.068 [0.048, 0.088] | 0.116 [0.081, 0.150] |
| CNN, attention pool | 0.407 [0.362, 0.464] | 0.316 [0.288, 0.348] | 0.520 [0.440, 0.640] | 0.293 [0.262, 0.324] | 0.076 [0.064, 0.091] | 0.132 [0.112, 0.159] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 9 | 16 | 119 | 106 |
| CNN, mean pool | 10 | 15 | 137 | 88 |
| CNN, attention pool | 13 | 12 | 159 | 66 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | 0.536 [0.480, 0.592] | 0.533 [0.462, 0.604] | 0.543 [0.444, 0.642] | 0.709 [0.654, 0.760] | 0.608 [0.549, 0.667] |
| CNN, mean pool | 0.499 [0.445, 0.550] | 0.488 [0.436, 0.536] | 0.467 [0.408, 0.521] | 0.531 [0.444, 0.617] | 0.675 [0.624, 0.723] | 0.552 [0.498, 0.601] |
| CNN, attention pool | 0.477 [0.435, 0.520] | 0.480 [0.440, 0.520] | 0.485 [0.444, 0.533] | 0.469 [0.395, 0.543] | 0.656 [0.619, 0.693] | 0.558 [0.519, 0.597] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 90 | 79 | 37 | 44 |
| CNN, mean pool | 79 | 90 | 38 | 43 |
| CNN, attention pool | 82 | 87 | 43 | 38 |
