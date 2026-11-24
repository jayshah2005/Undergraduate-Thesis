# Disease: whole-leaf CNN on Baco + Chardonnay + Vidal, full spectrum

Every model uses SNV and trains on the 3 training rows; the CNN picks its checkpoint on the validation row. Balanced accuracy (0.5 = chance); a leaf counts as positive at P ≥ 0.5; 95% bootstrap interval. CNN = mean probability of 3 seeds. Δ: paired difference to logistic regression. Pooled over varieties, so recognising a variety can raise it; see the per-variety table. Every metric: `metrics.md`.

## Leafroll (255 + / 493 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.572 [0.540, 0.604] | — | 0.568 | 0.560 | 0.692 | 0.478 | 0.594 |
| small ResNet, mean pool | 0.585 [0.564, 0.605] | +0.014 [-0.022, +0.049] | 0.572 | 0.643 | 0.548 | 0.540 | 0.634 |
| small ResNet, attention pool | 0.612 [0.588, 0.636] | +0.040 [+0.002, +0.075] | 0.489 | 0.668 | 0.611 | 0.522 | 0.645 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.471 | 0.478 | 0.522 |
| small ResNet, mean pool | 0.503 | 0.442 | 0.505 |
| small ResNet, attention pool | 0.518 | 0.520 | 0.502 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.584, 0.575, 0.559 | 0.573 ± 0.012 |
| small ResNet, attention pool | 0.617, 0.609, 0.601 | 0.609 ± 0.008 |

Median best epoch: small ResNet, mean pool 17, small ResNet, attention pool 18.

## Red blotch (396 + / 352 −)

| model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) | Fold 2 (R2) | Fold 3 (R3) | Fold 4 (R4) | Fold 5 (R5) |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.600 [0.573, 0.626] | — | 0.576 | 0.624 | 0.663 | 0.521 | 0.604 |
| small ResNet, mean pool | 0.595 [0.571, 0.619] | -0.004 [-0.038, +0.028] | 0.558 | 0.586 | 0.623 | 0.571 | 0.630 |
| small ResNet, attention pool | 0.594 [0.571, 0.616] | -0.006 [-0.038, +0.025] | 0.522 | 0.626 | 0.599 | 0.578 | 0.650 |
| small ResNet, mean ∥ max pool | 0.619 [0.598, 0.640] | +0.020 [-0.012, +0.051] | 0.602 | 0.599 | 0.664 | 0.543 | 0.653 |

Fold k tests on row Rk of every variety (3 field rows, since each variety has its own rows R1–R5); the next row (R1 after R5) is validation and the other 3 train.

Balanced accuracy per variety:

| model | Baco | Chardonnay | Vidal |
|---|---|---|---|
| logistic regression | 0.462 | 0.542 | 0.545 |
| small ResNet, mean pool | 0.481 | 0.545 | 0.553 |
| small ResNet, attention pool | 0.487 | 0.541 | 0.542 |
| small ResNet, mean ∥ max pool | 0.514 | 0.557 | 0.564 |

Single seeds (the ensemble above averages their probabilities):

| model | seeds | mean ± SD |
|---|---|---|
| small ResNet, mean pool | 0.578, 0.623, 0.572 | 0.591 ± 0.028 |
| small ResNet, attention pool | 0.591, 0.592, 0.597 | 0.593 ± 0.004 |
| small ResNet, mean ∥ max pool | 0.576, 0.612, 0.616 | 0.601 ± 0.022 |

Median best epoch: small ResNet, mean pool 10, small ResNet, attention pool 13, small ResNet, mean ∥ max pool 11.

Each change against the same model without it (ECA, CBAM: same backbone and pooling without the module; mean ∥ max: the same backbone's mean pool); paired, on the same resamples:

| model | compared with | Δ bal. acc. [95% CI] |
|---|---|---|
| small ResNet, mean ∥ max pool | small ResNet, mean pool | +0.024 [+0.003, +0.047] |
