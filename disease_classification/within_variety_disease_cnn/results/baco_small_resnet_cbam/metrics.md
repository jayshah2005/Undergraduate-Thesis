# Disease: whole-leaf CNN on Baco, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | 0.528 [0.468, 0.589] | 0.495 [0.402, 0.589] | 0.553 [0.475, 0.638] | 0.457 [0.393, 0.523] | 0.475 [0.400, 0.546] |
| small ResNet + CBAM, mean pool | 0.556 [0.556, 0.556] | 0.569 [0.569, 0.569] | 0.467 [0.467, 0.467] | 0.645 [0.645, 0.645] | 0.500 [0.500, 0.500] | 0.483 [0.483, 0.483] |
| small ResNet + CBAM, attention pool | 0.559 [0.526, 0.596] | 0.560 [0.528, 0.597] | 0.551 [0.505, 0.607] | 0.567 [0.518, 0.617] | 0.492 [0.458, 0.529] | 0.520 [0.482, 0.560] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 53 | 54 | 63 | 78 |
| small ResNet + CBAM, mean pool | 50 | 57 | 50 | 91 |
| small ResNet + CBAM, attention pool | 59 | 48 | 61 | 80 |

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | 0.508 [0.448, 0.569] | 0.533 [0.452, 0.615] | 0.478 [0.389, 0.566] | 0.550 [0.492, 0.604] | 0.541 [0.474, 0.604] |
| small ResNet + CBAM, mean pool | 0.457 [0.422, 0.495] | 0.444 [0.407, 0.480] | 0.304 [0.259, 0.356] | 0.611 [0.558, 0.664] | 0.482 [0.430, 0.537] | 0.373 [0.324, 0.422] |
| small ResNet + CBAM, attention pool | 0.497 [0.447, 0.547] | 0.492 [0.440, 0.540] | 0.437 [0.363, 0.511] | 0.558 [0.487, 0.628] | 0.541 [0.483, 0.597] | 0.484 [0.418, 0.545] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 72 | 63 | 59 | 54 |
| small ResNet + CBAM, mean pool | 41 | 94 | 44 | 69 |
| small ResNet + CBAM, attention pool | 59 | 76 | 50 | 63 |
