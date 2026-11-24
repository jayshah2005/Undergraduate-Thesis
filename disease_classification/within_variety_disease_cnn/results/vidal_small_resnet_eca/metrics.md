# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| small ResNet + ECA, mean pool | 0.419 [0.348, 0.488] | 0.416 [0.372, 0.460] | 0.423 [0.308, 0.538] | 0.415 [0.370, 0.460] | 0.077 [0.054, 0.100] | 0.131 [0.092, 0.169] |
| small ResNet + ECA, attention pool | 0.499 [0.453, 0.552] | 0.528 [0.492, 0.568] | 0.462 [0.385, 0.577] | 0.536 [0.496, 0.576] | 0.103 [0.085, 0.125] | 0.169 [0.140, 0.204] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| small ResNet + ECA, mean pool | 11 | 15 | 131 | 93 |
| small ResNet + ECA, attention pool | 12 | 14 | 104 | 120 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| small ResNet + ECA, mean pool | 0.547 [0.496, 0.596] | 0.532 [0.480, 0.580] | 0.509 [0.445, 0.567] | 0.584 [0.506, 0.662] | 0.733 [0.688, 0.777] | 0.601 [0.545, 0.649] |
| small ResNet + ECA, attention pool | 0.485 [0.438, 0.530] | 0.452 [0.408, 0.496] | 0.399 [0.347, 0.457] | 0.571 [0.494, 0.637] | 0.676 [0.626, 0.723] | 0.502 [0.449, 0.553] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| small ResNet + ECA, mean pool | 88 | 85 | 32 | 45 |
| small ResNet + ECA, attention pool | 69 | 104 | 33 | 44 |
