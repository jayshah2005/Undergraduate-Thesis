# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| CNN + ECA, mean pool | 0.543 [0.518, 0.571] | 0.508 [0.484, 0.536] | 0.794 [0.766, 0.822] | 0.291 [0.255, 0.333] | 0.459 [0.443, 0.478] | 0.582 [0.563, 0.605] |
| CNN + ECA, attention pool | 0.588 [0.569, 0.607] | 0.573 [0.552, 0.593] | 0.701 [0.673, 0.720] | 0.475 [0.447, 0.504] | 0.503 [0.487, 0.521] | 0.586 [0.568, 0.602] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| CNN + ECA, mean pool | 85 | 22 | 100 | 41 |
| CNN + ECA, attention pool | 75 | 32 | 74 | 67 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| CNN + ECA, mean pool | 0.472 [0.424, 0.521] | 0.476 [0.427, 0.524] | 0.519 [0.459, 0.585] | 0.425 [0.354, 0.496] | 0.519 [0.474, 0.564] | 0.519 [0.468, 0.571] |
| CNN + ECA, attention pool | 0.517 [0.480, 0.554] | 0.512 [0.476, 0.548] | 0.459 [0.407, 0.511] | 0.575 [0.522, 0.628] | 0.564 [0.522, 0.605] | 0.506 [0.460, 0.548] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| CNN + ECA, mean pool | 70 | 65 | 65 | 48 |
| CNN + ECA, attention pool | 62 | 73 | 48 | 65 |
