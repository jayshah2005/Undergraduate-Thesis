# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| CNN + CBAM, mean pool | 0.511 [0.466, 0.558] | 0.512 [0.464, 0.560] | 0.505 [0.449, 0.570] | 0.518 [0.447, 0.589] | 0.443 [0.398, 0.491] | 0.472 [0.425, 0.519] |
| CNN + CBAM, attention pool | 0.557 [0.519, 0.594] | 0.552 [0.516, 0.589] | 0.589 [0.533, 0.645] | 0.525 [0.482, 0.567] | 0.485 [0.450, 0.520] | 0.532 [0.487, 0.571] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| CNN + CBAM, mean pool | 54 | 53 | 68 | 73 |
| CNN + CBAM, attention pool | 63 | 44 | 67 | 74 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| CNN + CBAM, mean pool | 0.566 [0.517, 0.615] | 0.552 [0.504, 0.601] | 0.415 [0.348, 0.489] | 0.717 [0.646, 0.788] | 0.636 [0.568, 0.703] | 0.502 [0.437, 0.570] |
| CNN + CBAM, attention pool | 0.539 [0.490, 0.589] | 0.528 [0.480, 0.581] | 0.422 [0.356, 0.496] | 0.655 [0.584, 0.726] | 0.594 [0.531, 0.656] | 0.494 [0.429, 0.557] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| CNN + CBAM, mean pool | 56 | 79 | 32 | 81 |
| CNN + CBAM, attention pool | 57 | 78 | 39 | 74 |
