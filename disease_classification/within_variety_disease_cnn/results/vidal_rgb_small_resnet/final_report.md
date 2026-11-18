# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (26 + / 224 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.439 [0.352, 0.520] | — | 0.483 | 0.420 | 0.358 | 0.452 | 0.635 |
| small ResNet, mean pool | 0.368 [0.323, 0.400] | -0.071 [-0.149, +0.009] | 0.487 | 0.375 | 0.500 | 0.381 | 0.500 |
| small ResNet, attention pool | 0.392 [0.347, 0.421] | -0.047 [-0.127, +0.037] | 0.475 | 0.386 | 0.500 | 0.500 | 0.510 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.416, 0.422, 0.404 | 0.414 ± 0.009 |
| small ResNet, attention pool | 0.383, 0.399, 0.376 | 0.386 ± 0.012 |

Median best epoch: small ResNet, mean pool 2, small ResNet, attention pool 2.

## Red blotch (173 + / 77 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.488, 0.614] | — | 0.539 | 0.557 | 0.565 | 0.530 | 0.566 |
| small ResNet, mean pool | 0.482 [0.435, 0.528] | -0.069 [-0.123, -0.017] | 0.535 | 0.564 | 0.481 | 0.474 | 0.514 |
| small ResNet, attention pool | 0.483 [0.437, 0.528] | -0.069 [-0.121, -0.016] | 0.522 | 0.552 | 0.523 | 0.456 | 0.485 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.475, 0.500, 0.447 | 0.474 ± 0.027 |
| small ResNet, attention pool | 0.503, 0.499, 0.451 | 0.484 ± 0.029 |

Median best epoch: small ResNet, mean pool 4, small ResNet, attention pool 5.
