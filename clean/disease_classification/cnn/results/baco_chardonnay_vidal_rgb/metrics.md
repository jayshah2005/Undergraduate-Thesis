# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, RGB bands only

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table.

## Macro average: leafroll and red blotch

The mean of the two balanced accuracies below (0.5 = chance); each disease counts equally. Its interval resamples leaves, keeping each row's mix of disease combinations.

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.574 [0.555, 0.593] | — | 0.603 | 0.543 | 0.622 | 0.542 | 0.607 |
| CNN, mean pool | 0.583 [0.564, 0.601] | +0.009 [-0.006, +0.024] | 0.585 | 0.597 | 0.610 | 0.548 | 0.610 |
| CNN, attention pool | 0.594 [0.575, 0.612] | +0.020 [+0.004, +0.036] | 0.598 | 0.590 | 0.609 | 0.578 | 0.618 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Macro average per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.446 | 0.519 | 0.510 |
| CNN, mean pool | 0.458 | 0.513 | 0.520 |
| CNN, attention pool | 0.511 | 0.494 | 0.532 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.582, 0.562, 0.576 | 0.574 ± 0.010 |
| CNN, attention pool | 0.583, 0.587, 0.589 | 0.586 ± 0.003 |

## Leafroll (254 + / 494 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.513, 0.575] | — | 0.585 | 0.529 | 0.558 | 0.544 | 0.553 |
| CNN, mean pool | 0.565 [0.535, 0.593] | +0.021 [-0.004, +0.045] | 0.557 | 0.627 | 0.546 | 0.546 | 0.578 |
| CNN, attention pool | 0.580 [0.552, 0.607] | +0.036 [+0.010, +0.064] | 0.583 | 0.627 | 0.568 | 0.559 | 0.574 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.459 | 0.428 | 0.469 |
| CNN, mean pool | 0.462 | 0.436 | 0.493 |
| CNN, attention pool | 0.540 | 0.423 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.578, 0.548, 0.556 | 0.561 ± 0.015 |
| CNN, attention pool | 0.569, 0.572, 0.589 | 0.576 ± 0.011 |

Median best epoch: CNN, mean pool 6, CNN, attention pool 2.

## Red blotch (392 + / 356 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.604 [0.579, 0.630] | — | 0.622 | 0.557 | 0.687 | 0.540 | 0.661 |
| CNN, mean pool | 0.602 [0.576, 0.628] | -0.003 [-0.021, +0.018] | 0.612 | 0.567 | 0.673 | 0.550 | 0.641 |
| CNN, attention pool | 0.607 [0.582, 0.633] | +0.003 [-0.015, +0.022] | 0.612 | 0.553 | 0.649 | 0.597 | 0.662 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.434 | 0.610 | 0.551 |
| CNN, mean pool | 0.455 | 0.591 | 0.546 |
| CNN, attention pool | 0.483 | 0.565 | 0.562 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| CNN, mean pool | 0.587, 0.576, 0.596 | 0.587 ± 0.010 |
| CNN, attention pool | 0.596, 0.603, 0.589 | 0.596 ± 0.007 |

Median best epoch: CNN, mean pool 8, CNN, attention pool 10.
