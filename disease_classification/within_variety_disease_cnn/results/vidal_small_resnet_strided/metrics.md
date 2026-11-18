# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| small ResNet (no max pool), mean pool | 0.448 [0.390, 0.501] | 0.528 [0.484, 0.568] | 0.346 [0.231, 0.423] | 0.549 [0.500, 0.594] | 0.082 [0.057, 0.104] | 0.132 [0.092, 0.168] |
| small ResNet (no max pool), attention pool | 0.488 [0.437, 0.536] | 0.388 [0.344, 0.432] | 0.615 [0.538, 0.692] | 0.362 [0.317, 0.406] | 0.101 [0.086, 0.115] | 0.173 [0.147, 0.197] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| small ResNet (no max pool), mean pool | 9 | 17 | 101 | 123 |
| small ResNet (no max pool), attention pool | 16 | 10 | 143 | 81 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| small ResNet (no max pool), mean pool | 0.509 [0.453, 0.562] | 0.500 [0.452, 0.544] | 0.486 [0.428, 0.538] | 0.532 [0.442, 0.623] | 0.700 [0.651, 0.748] | 0.573 [0.523, 0.620] |
| small ResNet (no max pool), attention pool | 0.519 [0.467, 0.570] | 0.484 [0.436, 0.532] | 0.428 [0.376, 0.486] | 0.610 [0.519, 0.701] | 0.712 [0.658, 0.765] | 0.534 [0.480, 0.585] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| small ResNet (no max pool), mean pool | 84 | 89 | 36 | 41 |
| small ResNet (no max pool), attention pool | 74 | 99 | 30 | 47 |
