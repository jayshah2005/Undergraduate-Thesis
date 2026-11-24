# Disease: whole-leaf CNN on Baco, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.418 [0.371, 0.465] | 0.415 [0.371, 0.464] | 0.439 [0.374, 0.505] | 0.397 [0.333, 0.461] | 0.356 [0.311, 0.399] | 0.393 [0.339, 0.443] |
| CNN + ECA, mean pool | 0.486 [0.459, 0.515] | 0.476 [0.448, 0.504] | 0.561 [0.523, 0.598] | 0.411 [0.376, 0.454] | 0.420 [0.396, 0.444] | 0.480 [0.452, 0.510] |
| CNN + ECA, attention pool | 0.489 [0.458, 0.519] | 0.484 [0.452, 0.516] | 0.523 [0.486, 0.561] | 0.454 [0.404, 0.504] | 0.421 [0.394, 0.449] | 0.467 [0.439, 0.498] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 47 | 60 | 85 | 56 |
| CNN + ECA, mean pool | 60 | 47 | 83 | 58 |
| CNN + ECA, attention pool | 56 | 51 | 77 | 64 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.543 [0.494, 0.595] | 0.544 [0.496, 0.597] | 0.556 [0.481, 0.630] | 0.531 [0.460, 0.602] | 0.586 [0.538, 0.636] | 0.570 [0.516, 0.625] |
| CNN + ECA, mean pool | 0.530 [0.478, 0.580] | 0.516 [0.464, 0.569] | 0.370 [0.296, 0.444] | 0.690 [0.611, 0.761] | 0.588 [0.512, 0.663] | 0.455 [0.380, 0.524] |
| CNN + ECA, attention pool | 0.528 [0.479, 0.578] | 0.516 [0.468, 0.565] | 0.393 [0.326, 0.467] | 0.664 [0.593, 0.735] | 0.582 [0.517, 0.650] | 0.469 [0.404, 0.533] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 75 | 60 | 53 | 60 |
| CNN + ECA, mean pool | 50 | 85 | 35 | 78 |
| CNN + ECA, attention pool | 53 | 82 | 38 | 75 |
