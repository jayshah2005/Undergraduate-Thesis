# Disease: whole-leaf CNN on Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.557 [0.486, 0.628] | — | 0.535 | 0.489 | 0.468 | 0.589 | 0.406 |
| ResNet-18, mean pool | 0.440 [0.365, 0.518] | -0.117 [-0.219, -0.014] | 0.575 | 0.538 | 0.585 | 0.491 | 0.135 |
| ResNet-18, attention pool | 0.440 [0.368, 0.517] | -0.117 [-0.216, -0.019] | 0.470 | 0.504 | 0.532 | 0.539 | 0.594 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.404, 0.414, 0.478 | 0.432 ± 0.040 |
| ResNet-18, attention pool | 0.430, 0.409, 0.435 | 0.425 ± 0.014 |

Median best epoch: ResNet-18, mean pool 6, ResNet-18, attention pool 7.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.547 [0.488, 0.606] | — | 0.450 | 0.632 | 0.602 | 0.535 | 0.553 |
| ResNet-18, mean pool | 0.533 [0.479, 0.586] | -0.015 [-0.095, +0.065] | 0.487 | 0.592 | 0.518 | 0.580 | 0.539 |
| ResNet-18, attention pool | 0.510 [0.462, 0.559] | -0.038 [-0.117, +0.043] | 0.590 | 0.500 | 0.518 | 0.575 | 0.586 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.547, 0.471, 0.495 | 0.504 ± 0.038 |
| ResNet-18, attention pool | 0.539, 0.488, 0.496 | 0.508 ± 0.027 |

Median best epoch: ResNet-18, mean pool 9, ResNet-18, attention pool 10.
