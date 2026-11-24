# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| CNN + CBAM, mean pool | 0.425 [0.365, 0.490] | 0.488 [0.448, 0.528] | 0.346 [0.231, 0.462] | 0.504 [0.464, 0.545] | 0.075 [0.051, 0.100] | 0.123 [0.083, 0.164] |
| CNN + CBAM, attention pool | 0.468 [0.412, 0.532] | 0.656 [0.620, 0.692] | 0.231 [0.115, 0.346] | 0.705 [0.670, 0.741] | 0.083 [0.045, 0.125] | 0.122 [0.064, 0.184] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| CNN + CBAM, mean pool | 9 | 17 | 111 | 113 |
| CNN + CBAM, attention pool | 6 | 20 | 66 | 158 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| CNN + CBAM, mean pool | 0.562 [0.514, 0.608] | 0.548 [0.500, 0.596] | 0.526 [0.468, 0.584] | 0.597 [0.532, 0.662] | 0.746 [0.704, 0.784] | 0.617 [0.564, 0.667] |
| CNN + CBAM, attention pool | 0.476 [0.433, 0.517] | 0.440 [0.400, 0.476] | 0.382 [0.341, 0.422] | 0.571 [0.494, 0.636] | 0.667 [0.620, 0.710] | 0.485 [0.442, 0.526] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| CNN + CBAM, mean pool | 91 | 82 | 31 | 46 |
| CNN + CBAM, attention pool | 66 | 107 | 33 | 44 |
