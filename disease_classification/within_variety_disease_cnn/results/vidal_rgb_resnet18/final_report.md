# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| ResNet-18, mean pool | 0.395 [0.339, 0.447] | -0.044 [-0.127, +0.033] | 0.458 | 0.398 | 0.564 | 0.500 | 0.542 |
| ResNet-18, attention pool | 0.378 [0.331, 0.430] | -0.061 [-0.163, +0.041] | 0.399 | 0.477 | 0.521 | 0.452 | 0.552 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.405, 0.334, 0.462 | 0.400 ± 0.064 |
| ResNet-18, attention pool | 0.418, 0.302, 0.442 | 0.387 ± 0.075 |

Median best epoch: ResNet-18, mean pool 3, ResNet-18, attention pool 4.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| ResNet-18, mean pool | 0.509 [0.451, 0.564] | -0.042 [-0.120, +0.029] | 0.539 | 0.500 | 0.572 | 0.497 | 0.586 |
| ResNet-18, attention pool | 0.511 [0.474, 0.549] | -0.040 [-0.103, +0.027] | 0.493 | 0.500 | 0.561 | 0.552 | 0.586 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.475, 0.472, 0.469 | 0.472 ± 0.003 |
| ResNet-18, attention pool | 0.480, 0.504, 0.434 | 0.473 ± 0.036 |

Median best epoch: ResNet-18, mean pool 11, ResNet-18, attention pool 11.
