# Disease: whole-leaf CNN on Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Every metric: `metrics.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.416 [0.338, 0.493] | — | 0.495 | 0.420 | 0.188 | 0.414 | 0.646 |
| ResNet-18, mean pool | 0.360 [0.313, 0.413] | -0.056 [-0.160, +0.056] | 0.271 | 0.477 | 0.562 | 0.500 | 0.573 |
| ResNet-18, attention pool | 0.384 [0.367, 0.402] | -0.031 [-0.113, +0.051] | 0.523 | 0.489 | 0.521 | 0.417 | 0.104 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.478, 0.347, 0.396 | 0.407 ± 0.066 |
| ResNet-18, attention pool | 0.431, 0.396, 0.462 | 0.430 ± 0.033 |

Median best epoch: ResNet-18, mean pool 2, ResNet-18, attention pool 4.

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.538 [0.474, 0.598] | — | 0.567 | 0.525 | 0.550 | 0.530 | 0.536 |
| ResNet-18, mean pool | 0.436 [0.386, 0.485] | -0.102 [-0.166, -0.042] | 0.452 | 0.500 | 0.475 | 0.411 | 0.583 |
| ResNet-18, attention pool | 0.471 [0.419, 0.525] | -0.067 [-0.132, -0.003] | 0.445 | 0.500 | 0.500 | 0.568 | 0.538 |

Fold k tests on row Rk of Vidal; the next row (R1 after R5) is validation and the other 3 train.

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| ResNet-18, mean pool | 0.454, 0.481, 0.438 | 0.457 ± 0.022 |
| ResNet-18, attention pool | 0.429, 0.459, 0.479 | 0.456 ± 0.025 |

Median best epoch: ResNet-18, mean pool 8, ResNet-18, attention pool 11.
