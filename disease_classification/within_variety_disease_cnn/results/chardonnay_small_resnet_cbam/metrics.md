# Disease: whole-leaf CNN on Chardonnay, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (122 + / 128 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.537 [0.477, 0.596] | 0.536 [0.476, 0.596] | 0.557 [0.475, 0.639] | 0.516 [0.430, 0.602] | 0.523 [0.465, 0.582] | 0.540 [0.476, 0.603] |
| small ResNet + CBAM, mean pool | 0.567 [0.539, 0.595] | 0.568 [0.540, 0.596] | 0.525 [0.492, 0.557] | 0.609 [0.562, 0.656] | 0.561 [0.530, 0.593] | 0.542 [0.515, 0.570] |
| small ResNet + CBAM, attention pool | 0.517 [0.482, 0.553] | 0.516 [0.480, 0.552] | 0.574 [0.533, 0.615] | 0.461 [0.398, 0.523] | 0.504 [0.472, 0.538] | 0.536 [0.506, 0.568] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 68 | 54 | 62 | 66 |
| small ResNet + CBAM, mean pool | 64 | 58 | 50 | 78 |
| small ResNet + CBAM, attention pool | 70 | 52 | 69 | 59 |

## Red blotch (88 + / 162 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.605 [0.547, 0.662] | 0.636 [0.584, 0.688] | 0.500 [0.409, 0.591] | 0.710 [0.642, 0.772] | 0.484 [0.412, 0.559] | 0.492 [0.414, 0.565] |
| small ResNet + CBAM, mean pool | 0.603 [0.551, 0.653] | 0.644 [0.596, 0.692] | 0.466 [0.386, 0.545] | 0.741 [0.685, 0.796] | 0.494 [0.422, 0.568] | 0.480 [0.409, 0.548] |
| small ResNet + CBAM, attention pool | 0.558 [0.507, 0.608] | 0.616 [0.572, 0.660] | 0.364 [0.273, 0.443] | 0.753 [0.698, 0.809] | 0.444 [0.364, 0.524] | 0.400 [0.316, 0.475] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 44 | 44 | 47 | 115 |
| small ResNet + CBAM, mean pool | 41 | 47 | 42 | 120 |
| small ResNet + CBAM, attention pool | 32 | 56 | 40 | 122 |
