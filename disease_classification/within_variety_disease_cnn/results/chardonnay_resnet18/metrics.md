# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| ResNet-18, mean pool | 0.451 [0.406, 0.498] | 0.448 [0.404, 0.496] | 0.557 [0.492, 0.623] | 0.344 [0.281, 0.398] | 0.447 [0.409, 0.486] | 0.496 [0.447, 0.545] |
| ResNet-18, attention pool | 0.518 [0.478, 0.558] | 0.516 [0.476, 0.556] | 0.607 [0.541, 0.664] | 0.430 [0.375, 0.477] | 0.503 [0.469, 0.536] | 0.550 [0.506, 0.591] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| ResNet-18, mean pool | 68 | 54 | 84 | 44 |
| ResNet-18, attention pool | 74 | 48 | 73 | 55 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| ResNet-18, mean pool | 0.538 [0.485, 0.588] | 0.552 [0.504, 0.604] | 0.489 [0.409, 0.568] | 0.586 [0.525, 0.648] | 0.391 [0.336, 0.447] | 0.434 [0.369, 0.492] |
| ResNet-18, attention pool | 0.497 [0.444, 0.550] | 0.496 [0.444, 0.548] | 0.500 [0.409, 0.591] | 0.494 [0.426, 0.556] | 0.349 [0.301, 0.397] | 0.411 [0.348, 0.469] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| ResNet-18, mean pool | 43 | 45 | 67 | 95 |
| ResNet-18, attention pool | 44 | 44 | 82 | 80 |
