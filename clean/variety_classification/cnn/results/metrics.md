# Variety CNN: metrics

Balanced accuracy, chance 0.25; held-out field rows; 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on a fourth: the CNN its checkpoint, logistic regression its C. CNN scores are 3-seed ensembles.

| model | balanced accuracy |
|---|---|
| CNN `all` | 0.898 (0.880–0.916) |
| CNN `rgb` | 0.824 (0.799–0.846) |
| logistic regression `all` | 0.874 (0.853–0.894) |
| logistic regression `rgb` | 0.503 (0.472–0.533) |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.

## Paired differences (same resampled leaves)

| comparison | Δ balanced accuracy [95% CI] |
|---|---|
| CNN: 204 bands − RGB | +0.074 [+0.049, +0.100] |
| Logistic regression: 204 bands − RGB | +0.371 [+0.341, +0.404] |
| 204 bands: CNN − logistic regression | +0.024 [+0.007, +0.041] |
| RGB: CNN − logistic regression | +0.321 [+0.286, +0.355] |

## CNN detail

| input | seeds | seed mean | R1 | R2 | R3 | R4 | R5 | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `all` | 0.857, 0.856, 0.914 | 0.875 | 0.595 | 0.975 | 0.975 | 0.975 | 0.970 | 11 |
| `rgb` | 0.780, 0.791, 0.788 | 0.786 | 0.730 | 0.870 | 0.905 | 0.820 | 0.795 | 49 |
