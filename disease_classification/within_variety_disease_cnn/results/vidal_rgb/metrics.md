# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | 0.452 [0.396, 0.508] | 0.423 [0.269, 0.577] | 0.455 [0.397, 0.513] | 0.083 [0.053, 0.111] | 0.138 [0.088, 0.185] |
| CNN, mean pool | 0.415 [0.337, 0.494] | 0.348 [0.300, 0.396] | 0.500 [0.346, 0.654] | 0.330 [0.277, 0.384] | 0.080 [0.057, 0.102] | 0.138 [0.098, 0.177] |
| CNN, attention pool | 0.410 [0.367, 0.461] | 0.308 [0.280, 0.336] | 0.538 [0.462, 0.654] | 0.281 [0.250, 0.312] | 0.080 [0.069, 0.094] | 0.139 [0.119, 0.163] |
| small ResNet, mean pool | 0.368 [0.323, 0.400] | 0.324 [0.288, 0.360] | 0.423 [0.346, 0.462] | 0.312 [0.272, 0.353] | 0.067 [0.054, 0.075] | 0.115 [0.094, 0.129] |
| small ResNet, attention pool | 0.392 [0.347, 0.421] | 0.368 [0.340, 0.396] | 0.423 [0.346, 0.462] | 0.362 [0.330, 0.393] | 0.071 [0.058, 0.079] | 0.122 [0.099, 0.136] |
| small ResNet (no max pool), mean pool | 0.431 [0.359, 0.507] | 0.316 [0.276, 0.356] | 0.577 [0.423, 0.731] | 0.286 [0.246, 0.326] | 0.086 [0.066, 0.106] | 0.149 [0.114, 0.184] |
| small ResNet (no max pool), attention pool | 0.411 [0.400, 0.424] | 0.432 [0.412, 0.456] | 0.385 [0.385, 0.385] | 0.438 [0.415, 0.464] | 0.074 [0.071, 0.077] | 0.123 [0.120, 0.128] |
| ResNet-18, mean pool | 0.395 [0.339, 0.447] | 0.464 [0.420, 0.504] | 0.308 [0.230, 0.385] | 0.482 [0.438, 0.527] | 0.065 [0.044, 0.084] | 0.107 [0.073, 0.139] |
| ResNet-18, attention pool | 0.378 [0.331, 0.430] | 0.464 [0.424, 0.504] | 0.269 [0.192, 0.346] | 0.487 [0.446, 0.527] | 0.057 [0.040, 0.077] | 0.095 [0.067, 0.126] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 11 | 15 | 122 | 102 |
| CNN, mean pool | 13 | 13 | 150 | 74 |
| CNN, attention pool | 14 | 12 | 161 | 63 |
| small ResNet, mean pool | 11 | 15 | 154 | 70 |
| small ResNet, attention pool | 11 | 15 | 143 | 81 |
| small ResNet (no max pool), mean pool | 15 | 11 | 160 | 64 |
| small ResNet (no max pool), attention pool | 10 | 16 | 126 | 98 |
| ResNet-18, mean pool | 8 | 18 | 116 | 108 |
| ResNet-18, attention pool | 7 | 19 | 115 | 109 |

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | 0.544 [0.484, 0.604] | 0.532 [0.462, 0.601] | 0.571 [0.468, 0.675] | 0.736 [0.681, 0.789] | 0.617 [0.556, 0.675] |
| CNN, mean pool | 0.482 [0.431, 0.531] | 0.488 [0.444, 0.532] | 0.497 [0.451, 0.549] | 0.468 [0.377, 0.545] | 0.677 [0.634, 0.718] | 0.573 [0.531, 0.616] |
| CNN, attention pool | 0.454 [0.408, 0.495] | 0.464 [0.424, 0.500] | 0.480 [0.439, 0.526] | 0.429 [0.351, 0.494] | 0.654 [0.616, 0.687] | 0.553 [0.515, 0.590] |
| small ResNet, mean pool | 0.482 [0.435, 0.528] | 0.468 [0.428, 0.508] | 0.445 [0.399, 0.491] | 0.519 [0.442, 0.597] | 0.675 [0.632, 0.718] | 0.537 [0.494, 0.578] |
| small ResNet, attention pool | 0.483 [0.437, 0.528] | 0.484 [0.444, 0.528] | 0.486 [0.434, 0.538] | 0.481 [0.403, 0.558] | 0.677 [0.637, 0.715] | 0.566 [0.521, 0.611] |
| small ResNet (no max pool), mean pool | 0.479 [0.428, 0.528] | 0.468 [0.424, 0.512] | 0.451 [0.393, 0.503] | 0.506 [0.429, 0.584] | 0.672 [0.626, 0.718] | 0.540 [0.489, 0.586] |
| small ResNet (no max pool), attention pool | 0.498 [0.446, 0.545] | 0.520 [0.476, 0.564] | 0.555 [0.503, 0.607] | 0.442 [0.364, 0.519] | 0.691 [0.651, 0.726] | 0.615 [0.570, 0.658] |
| ResNet-18, mean pool | 0.509 [0.451, 0.564] | 0.560 [0.508, 0.612] | 0.642 [0.578, 0.699] | 0.377 [0.286, 0.468] | 0.698 [0.660, 0.735] | 0.669 [0.621, 0.712] |
| ResNet-18, attention pool | 0.511 [0.474, 0.549] | 0.488 [0.452, 0.524] | 0.451 [0.405, 0.497] | 0.571 [0.506, 0.623] | 0.703 [0.667, 0.739] | 0.549 [0.507, 0.590] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 92 | 81 | 33 | 44 |
| CNN, mean pool | 86 | 87 | 41 | 36 |
| CNN, attention pool | 83 | 90 | 44 | 33 |
| small ResNet, mean pool | 77 | 96 | 37 | 40 |
| small ResNet, attention pool | 84 | 89 | 40 | 37 |
| small ResNet (no max pool), mean pool | 78 | 95 | 38 | 39 |
| small ResNet (no max pool), attention pool | 96 | 77 | 43 | 34 |
| ResNet-18, mean pool | 111 | 62 | 48 | 29 |
| ResNet-18, attention pool | 78 | 95 | 33 | 44 |
