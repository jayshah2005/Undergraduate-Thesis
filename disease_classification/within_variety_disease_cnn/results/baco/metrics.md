# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| CNN, mean pool | 0.546 [0.522, 0.571] | 0.552 [0.528, 0.577] | 0.495 [0.467, 0.523] | 0.596 [0.560, 0.631] | 0.482 [0.455, 0.510] | 0.488 [0.466, 0.516] |
| CNN, attention pool | 0.529 [0.502, 0.556] | 0.532 [0.504, 0.560] | 0.505 [0.477, 0.542] | 0.553 [0.511, 0.596] | 0.462 [0.433, 0.491] | 0.482 [0.457, 0.511] |
| small ResNet, mean pool | 0.549 [0.533, 0.566] | 0.560 [0.544, 0.577] | 0.467 [0.439, 0.495] | 0.631 [0.610, 0.645] | 0.490 [0.471, 0.510] | 0.478 [0.457, 0.500] |
| small ResNet, attention pool | 0.576 [0.545, 0.610] | 0.577 [0.548, 0.609] | 0.570 [0.523, 0.626] | 0.582 [0.539, 0.617] | 0.508 [0.478, 0.540] | 0.537 [0.500, 0.576] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| CNN, mean pool | 53 | 54 | 57 | 84 |
| CNN, attention pool | 54 | 53 | 63 | 78 |
| small ResNet, mean pool | 50 | 57 | 52 | 89 |
| small ResNet, attention pool | 61 | 46 | 59 | 82 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| CNN, mean pool | 0.504 [0.456, 0.550] | 0.492 [0.444, 0.536] | 0.363 [0.296, 0.430] | 0.646 [0.575, 0.717] | 0.551 [0.483, 0.614] | 0.438 [0.371, 0.496] |
| CNN, attention pool | 0.505 [0.465, 0.547] | 0.496 [0.456, 0.536] | 0.407 [0.356, 0.459] | 0.602 [0.540, 0.664] | 0.550 [0.500, 0.604] | 0.468 [0.419, 0.515] |
| small ResNet, mean pool | 0.525 [0.480, 0.569] | 0.520 [0.476, 0.565] | 0.474 [0.407, 0.541] | 0.575 [0.504, 0.637] | 0.571 [0.523, 0.619] | 0.518 [0.463, 0.570] |
| small ResNet, attention pool | 0.511 [0.468, 0.553] | 0.508 [0.464, 0.552] | 0.481 [0.422, 0.541] | 0.540 [0.469, 0.602] | 0.556 [0.509, 0.600] | 0.516 [0.465, 0.566] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| CNN, mean pool | 49 | 86 | 40 | 73 |
| CNN, attention pool | 55 | 80 | 45 | 68 |
| small ResNet, mean pool | 64 | 71 | 48 | 65 |
| small ResNet, attention pool | 65 | 70 | 52 | 61 |
