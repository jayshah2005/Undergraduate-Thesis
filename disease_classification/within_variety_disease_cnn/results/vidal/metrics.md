# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | 0.632 [0.588, 0.676] | 0.462 [0.346, 0.577] | 0.652 [0.607, 0.696] | 0.133 [0.097, 0.171] | 0.207 [0.151, 0.264] |
| CNN, mean pool | 0.474 [0.419, 0.525] | 0.576 [0.536, 0.616] | 0.346 [0.231, 0.423] | 0.603 [0.558, 0.647] | 0.092 [0.065, 0.116] | 0.145 [0.102, 0.183] |
| CNN, attention pool | 0.447 [0.387, 0.510] | 0.496 [0.448, 0.544] | 0.385 [0.269, 0.500] | 0.509 [0.460, 0.558] | 0.083 [0.060, 0.108] | 0.137 [0.098, 0.177] |
| small ResNet, mean pool | 0.513 [0.441, 0.589] | 0.524 [0.480, 0.572] | 0.500 [0.385, 0.654] | 0.527 [0.482, 0.576] | 0.109 [0.081, 0.139] | 0.179 [0.133, 0.228] |
| small ResNet, attention pool | 0.432 [0.377, 0.488] | 0.500 [0.456, 0.548] | 0.346 [0.231, 0.423] | 0.518 [0.469, 0.567] | 0.077 [0.056, 0.099] | 0.126 [0.091, 0.161] |
| small ResNet (no max pool), mean pool | 0.448 [0.390, 0.501] | 0.528 [0.484, 0.568] | 0.346 [0.231, 0.423] | 0.549 [0.500, 0.594] | 0.082 [0.057, 0.104] | 0.132 [0.092, 0.168] |
| small ResNet (no max pool), attention pool | 0.488 [0.437, 0.536] | 0.388 [0.344, 0.432] | 0.615 [0.538, 0.692] | 0.362 [0.317, 0.406] | 0.101 [0.086, 0.115] | 0.173 [0.147, 0.197] |
| ResNet-18, mean pool | 0.440 [0.365, 0.518] | 0.544 [0.492, 0.592] | 0.308 [0.192, 0.462] | 0.571 [0.518, 0.621] | 0.077 [0.044, 0.111] | 0.123 [0.072, 0.179] |
| ResNet-18, attention pool | 0.440 [0.368, 0.517] | 0.484 [0.440, 0.528] | 0.385 [0.269, 0.538] | 0.496 [0.446, 0.540] | 0.081 [0.055, 0.110] | 0.134 [0.091, 0.183] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 12 | 14 | 78 | 146 |
| CNN, mean pool | 9 | 17 | 89 | 135 |
| CNN, attention pool | 10 | 16 | 110 | 114 |
| small ResNet, mean pool | 13 | 13 | 106 | 118 |
| small ResNet, attention pool | 9 | 17 | 108 | 116 |
| small ResNet (no max pool), mean pool | 9 | 17 | 101 | 123 |
| small ResNet (no max pool), attention pool | 16 | 10 | 143 | 81 |
| ResNet-18, mean pool | 8 | 18 | 96 | 128 |
| ResNet-18, attention pool | 10 | 16 | 113 | 111 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | 0.608 [0.552, 0.660] | 0.705 [0.642, 0.769] | 0.390 [0.286, 0.481] | 0.722 [0.684, 0.760] | 0.713 [0.669, 0.756] |
| CNN, mean pool | 0.557 [0.498, 0.612] | 0.516 [0.460, 0.568] | 0.451 [0.387, 0.514] | 0.662 [0.571, 0.753] | 0.750 [0.690, 0.808] | 0.563 [0.502, 0.620] |
| CNN, attention pool | 0.492 [0.442, 0.541] | 0.452 [0.408, 0.496] | 0.387 [0.335, 0.439] | 0.597 [0.519, 0.675] | 0.684 [0.629, 0.736] | 0.494 [0.441, 0.545] |
| small ResNet, mean pool | 0.503 [0.450, 0.556] | 0.492 [0.444, 0.540] | 0.474 [0.416, 0.532] | 0.532 [0.442, 0.623] | 0.695 [0.647, 0.743] | 0.564 [0.512, 0.613] |
| small ResNet, attention pool | 0.509 [0.456, 0.560] | 0.500 [0.452, 0.548] | 0.486 [0.428, 0.543] | 0.532 [0.455, 0.610] | 0.700 [0.654, 0.746] | 0.573 [0.518, 0.623] |
| small ResNet (no max pool), mean pool | 0.509 [0.453, 0.562] | 0.500 [0.452, 0.544] | 0.486 [0.428, 0.538] | 0.532 [0.442, 0.623] | 0.700 [0.651, 0.748] | 0.573 [0.523, 0.620] |
| small ResNet (no max pool), attention pool | 0.519 [0.467, 0.570] | 0.484 [0.436, 0.532] | 0.428 [0.376, 0.486] | 0.610 [0.519, 0.701] | 0.712 [0.658, 0.765] | 0.534 [0.480, 0.585] |
| ResNet-18, mean pool | 0.533 [0.479, 0.586] | 0.508 [0.456, 0.560] | 0.468 [0.410, 0.532] | 0.597 [0.506, 0.688] | 0.723 [0.672, 0.775] | 0.568 [0.513, 0.623] |
| ResNet-18, attention pool | 0.510 [0.462, 0.559] | 0.516 [0.472, 0.560] | 0.526 [0.474, 0.578] | 0.494 [0.416, 0.571] | 0.700 [0.661, 0.740] | 0.601 [0.556, 0.645] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 51 | 47 | 30 |
| CNN, mean pool | 78 | 95 | 26 | 51 |
| CNN, attention pool | 67 | 106 | 31 | 46 |
| small ResNet, mean pool | 82 | 91 | 36 | 41 |
| small ResNet, attention pool | 84 | 89 | 36 | 41 |
| small ResNet (no max pool), mean pool | 84 | 89 | 36 | 41 |
| small ResNet (no max pool), attention pool | 74 | 99 | 30 | 47 |
| ResNet-18, mean pool | 81 | 92 | 31 | 46 |
| ResNet-18, attention pool | 91 | 82 | 39 | 38 |
