# Disease: whole-leaf CNN on Baco, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (107 + / 141 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.524 [0.464, 0.587] | — | 0.534 | 0.585 | 0.511 | 0.562 | 0.486 |
| ResNet-18, mean pool | 0.566 [0.526, 0.608] | +0.042 [-0.034, +0.120] | 0.526 | 0.507 | 0.500 | 0.513 | 0.446 |
| ResNet-18, attention pool | 0.533 [0.487, 0.578] | +0.009 [-0.068, +0.086] | 0.583 | 0.457 | 0.500 | 0.488 | 0.398 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.523, 0.520, 0.523 | 0.522 ± 0.002 |
| ResNet-18, attention pool | 0.487, 0.541, 0.510 | 0.513 ± 0.027 |

Median best epoch: ResNet-18, mean pool 9, ResNet-18, attention pool 9.

## Red blotch (135 + / 113 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.506 [0.445, 0.565] | — | 0.483 | 0.528 | 0.462 | 0.560 | 0.465 |
| ResNet-18, mean pool | 0.505 [0.448, 0.559] | -0.001 [-0.075, +0.080] | 0.451 | 0.463 | 0.566 | 0.540 | 0.441 |
| ResNet-18, attention pool | 0.471 [0.425, 0.522] | -0.034 [-0.101, +0.040] | 0.464 | 0.518 | 0.444 | 0.500 | 0.389 |

Fold k tests on row Rk of Baco; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.522, 0.493, 0.506 | 0.507 ± 0.015 |
| ResNet-18, attention pool | 0.502, 0.500, 0.489 | 0.497 ± 0.007 |

Median best epoch: ResNet-18, mean pool 7, ResNet-18, attention pool 14.
