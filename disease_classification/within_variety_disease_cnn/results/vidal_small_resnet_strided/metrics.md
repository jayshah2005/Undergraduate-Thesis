# Disease: whole-leaf CNN on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | 0.660 [0.616, 0.700] | 0.400 [0.320, 0.480] | 0.689 [0.640, 0.733] | 0.125 [0.096, 0.157] | 0.190 [0.147, 0.235] |
| small ResNet (no max pool), mean pool | 0.433 [0.384, 0.471] | 0.460 [0.420, 0.504] | 0.400 [0.320, 0.440] | 0.467 [0.422, 0.511] | 0.077 [0.060, 0.089] | 0.129 [0.101, 0.149] |
| small ResNet (no max pool), attention pool | 0.493 [0.420, 0.567] | 0.568 [0.512, 0.620] | 0.400 [0.280, 0.520] | 0.587 [0.529, 0.644] | 0.097 [0.065, 0.130] | 0.156 [0.105, 0.208] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 10 | 15 | 70 | 155 |
| small ResNet (no max pool), mean pool | 10 | 15 | 120 | 105 |
| small ResNet (no max pool), attention pool | 10 | 15 | 93 | 132 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | 0.612 [0.560, 0.664] | 0.722 [0.651, 0.787] | 0.383 [0.284, 0.481] | 0.709 [0.672, 0.746] | 0.716 [0.669, 0.758] |
| small ResNet (no max pool), mean pool | 0.512 [0.467, 0.558] | 0.488 [0.448, 0.532] | 0.444 [0.396, 0.491] | 0.580 [0.506, 0.654] | 0.688 [0.643, 0.735] | 0.540 [0.496, 0.584] |
| small ResNet (no max pool), attention pool | 0.506 [0.451, 0.558] | 0.484 [0.440, 0.532] | 0.444 [0.391, 0.497] | 0.568 [0.481, 0.655] | 0.682 [0.629, 0.735] | 0.538 [0.489, 0.585] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 47 | 50 | 31 |
| small ResNet (no max pool), mean pool | 75 | 94 | 34 | 47 |
| small ResNet (no max pool), attention pool | 75 | 94 | 35 | 46 |
