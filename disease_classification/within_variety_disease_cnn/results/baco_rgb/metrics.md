# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| CNN, mean pool | 0.370 [0.344, 0.398] | 0.383 [0.355, 0.411] | 0.271 [0.243, 0.299] | 0.468 [0.426, 0.518] | 0.279 [0.250, 0.308] | 0.275 [0.246, 0.303] |
| CNN, attention pool | 0.440 [0.407, 0.472] | 0.431 [0.399, 0.464] | 0.505 [0.467, 0.542] | 0.376 [0.326, 0.426] | 0.380 [0.352, 0.407] | 0.434 [0.402, 0.462] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| CNN, mean pool | 29 | 78 | 75 | 66 |
| CNN, attention pool | 54 | 53 | 88 | 53 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| CNN, mean pool | 0.525 [0.475, 0.576] | 0.508 [0.456, 0.557] | 0.333 [0.267, 0.400] | 0.717 [0.637, 0.788] | 0.584 [0.500, 0.667] | 0.425 [0.351, 0.493] |
| CNN, attention pool | 0.548 [0.500, 0.595] | 0.528 [0.480, 0.577] | 0.326 [0.259, 0.393] | 0.770 [0.699, 0.832] | 0.629 [0.544, 0.714] | 0.429 [0.357, 0.498] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| CNN, mean pool | 45 | 90 | 32 | 81 |
| CNN, attention pool | 44 | 91 | 26 | 87 |
