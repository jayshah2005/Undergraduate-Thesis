# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| small ResNet + CBAM, mean pool | 0.451 [0.387, 0.507] | 0.352 [0.312, 0.392] | 0.577 [0.462, 0.692] | 0.326 [0.286, 0.366] | 0.090 [0.072, 0.106] | 0.156 [0.125, 0.183] |
| small ResNet + CBAM, attention pool | 0.453 [0.406, 0.507] | 0.264 [0.244, 0.284] | 0.692 [0.615, 0.808] | 0.214 [0.196, 0.237] | 0.093 [0.082, 0.106] | 0.164 [0.144, 0.187] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| small ResNet + CBAM, mean pool | 15 | 11 | 151 | 73 |
| small ResNet + CBAM, attention pool | 18 | 8 | 176 | 48 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| small ResNet + CBAM, mean pool | 0.521 [0.480, 0.562] | 0.512 [0.476, 0.548] | 0.497 [0.451, 0.543] | 0.545 [0.481, 0.610] | 0.711 [0.675, 0.748] | 0.585 [0.544, 0.622] |
| small ResNet + CBAM, attention pool | 0.509 [0.469, 0.547] | 0.480 [0.440, 0.520] | 0.434 [0.382, 0.480] | 0.584 [0.519, 0.649] | 0.701 [0.660, 0.738] | 0.536 [0.489, 0.580] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| small ResNet + CBAM, mean pool | 86 | 87 | 35 | 42 |
| small ResNet + CBAM, attention pool | 75 | 98 | 32 | 45 |
