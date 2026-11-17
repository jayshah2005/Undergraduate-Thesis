# Disease: whole-leaf CNN on Vidal, RGB bands only: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | 0.460 [0.404, 0.516] | 0.360 [0.200, 0.520] | 0.471 [0.409, 0.529] | 0.070 [0.042, 0.098] | 0.118 [0.069, 0.164] |
| CNN, mean pool | 0.396 [0.327, 0.460] | 0.392 [0.344, 0.436] | 0.400 [0.280, 0.520] | 0.391 [0.342, 0.440] | 0.068 [0.048, 0.088] | 0.116 [0.081, 0.150] |
| CNN, attention pool | 0.407 [0.362, 0.464] | 0.316 [0.288, 0.348] | 0.520 [0.440, 0.640] | 0.293 [0.262, 0.324] | 0.076 [0.064, 0.091] | 0.132 [0.112, 0.159] |
| small ResNet, mean pool | 0.376 [0.320, 0.422] | 0.388 [0.352, 0.424] | 0.360 [0.280, 0.440] | 0.391 [0.351, 0.431] | 0.062 [0.046, 0.076] | 0.105 [0.079, 0.129] |
| small ResNet, attention pool | 0.340 [0.291, 0.376] | 0.356 [0.320, 0.392] | 0.320 [0.240, 0.360] | 0.360 [0.320, 0.400] | 0.053 [0.039, 0.061] | 0.090 [0.067, 0.105] |
| small ResNet (no max pool), mean pool | 0.353 [0.284, 0.433] | 0.380 [0.336, 0.424] | 0.320 [0.200, 0.480] | 0.387 [0.338, 0.431] | 0.055 [0.034, 0.079] | 0.094 [0.058, 0.136] |
| small ResNet (no max pool), attention pool | 0.458 [0.442, 0.473] | 0.472 [0.444, 0.500] | 0.440 [0.440, 0.440] | 0.476 [0.444, 0.507] | 0.085 [0.081, 0.090] | 0.143 [0.137, 0.150] |
| ResNet-18, mean pool | 0.360 [0.313, 0.413] | 0.456 [0.420, 0.496] | 0.240 [0.160, 0.360] | 0.480 [0.440, 0.524] | 0.049 [0.032, 0.068] | 0.081 [0.054, 0.113] |
| ResNet-18, attention pool | 0.384 [0.367, 0.402] | 0.404 [0.372, 0.436] | 0.360 [0.360, 0.360] | 0.409 [0.373, 0.444] | 0.063 [0.060, 0.067] | 0.108 [0.103, 0.113] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 9 | 16 | 119 | 106 |
| CNN, mean pool | 10 | 15 | 137 | 88 |
| CNN, attention pool | 13 | 12 | 159 | 66 |
| small ResNet, mean pool | 9 | 16 | 137 | 88 |
| small ResNet, attention pool | 8 | 17 | 144 | 81 |
| small ResNet (no max pool), mean pool | 8 | 17 | 138 | 87 |
| small ResNet (no max pool), attention pool | 11 | 14 | 118 | 107 |
| ResNet-18, mean pool | 6 | 19 | 117 | 108 |
| ResNet-18, attention pool | 9 | 16 | 133 | 92 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | 0.536 [0.480, 0.592] | 0.533 [0.462, 0.604] | 0.543 [0.444, 0.642] | 0.709 [0.654, 0.760] | 0.608 [0.549, 0.667] |
| CNN, mean pool | 0.499 [0.445, 0.550] | 0.488 [0.436, 0.536] | 0.467 [0.408, 0.521] | 0.531 [0.444, 0.617] | 0.675 [0.624, 0.723] | 0.552 [0.498, 0.601] |
| CNN, attention pool | 0.477 [0.435, 0.520] | 0.480 [0.440, 0.520] | 0.485 [0.444, 0.533] | 0.469 [0.395, 0.543] | 0.656 [0.619, 0.693] | 0.558 [0.519, 0.597] |
| small ResNet, mean pool | 0.484 [0.441, 0.526] | 0.480 [0.444, 0.520] | 0.473 [0.432, 0.515] | 0.494 [0.420, 0.568] | 0.661 [0.624, 0.698] | 0.552 [0.514, 0.592] |
| small ResNet, attention pool | 0.510 [0.459, 0.564] | 0.512 [0.464, 0.560] | 0.515 [0.456, 0.574] | 0.506 [0.420, 0.593] | 0.685 [0.641, 0.731] | 0.588 [0.537, 0.636] |
| small ResNet (no max pool), mean pool | 0.503 [0.449, 0.557] | 0.476 [0.424, 0.528] | 0.426 [0.367, 0.491] | 0.580 [0.494, 0.667] | 0.679 [0.624, 0.737] | 0.524 [0.464, 0.582] |
| small ResNet (no max pool), attention pool | 0.480 [0.438, 0.523] | 0.480 [0.440, 0.520] | 0.479 [0.426, 0.533] | 0.481 [0.420, 0.543] | 0.659 [0.620, 0.696] | 0.555 [0.509, 0.599] |
| ResNet-18, mean pool | 0.436 [0.386, 0.485] | 0.424 [0.380, 0.468] | 0.402 [0.349, 0.456] | 0.469 [0.383, 0.543] | 0.613 [0.562, 0.661] | 0.486 [0.433, 0.534] |
| ResNet-18, attention pool | 0.471 [0.419, 0.525] | 0.476 [0.428, 0.528] | 0.485 [0.426, 0.544] | 0.457 [0.370, 0.543] | 0.651 [0.603, 0.697] | 0.556 [0.504, 0.605] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 90 | 79 | 37 | 44 |
| CNN, mean pool | 79 | 90 | 38 | 43 |
| CNN, attention pool | 82 | 87 | 43 | 38 |
| small ResNet, mean pool | 80 | 89 | 41 | 40 |
| small ResNet, attention pool | 87 | 82 | 40 | 41 |
| small ResNet (no max pool), mean pool | 72 | 97 | 34 | 47 |
| small ResNet (no max pool), attention pool | 81 | 88 | 42 | 39 |
| ResNet-18, mean pool | 68 | 101 | 43 | 38 |
| ResNet-18, attention pool | 82 | 87 | 44 | 37 |
