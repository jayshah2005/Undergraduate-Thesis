# Variety CNN: metrics, trained without augmentation

Balanced accuracy, chance 0.25; held-out field rows; 95% bootstrap interval. Every model uses SNV, trains on 3 rows and is selected on a fourth: the CNN its checkpoint, logistic regression its C. CNN scores are 3-seed ensembles.

| model | balanced accuracy |
|---|---|
| CNN `all` | 0.914 (0.898–0.931) |
| CNN `rgb` | 0.928 (0.911–0.943) |
| logistic regression `all` | 0.874 (0.853–0.894) |
| logistic regression `rgb` | 0.503 (0.472–0.533) |

Leaf size alone (pixel count, height, width), logistic regression: 0.642.

## Paired differences (same resampled leaves)

| comparison | Δ balanced accuracy [95% CI] |
|---|---|
| CNN: 204 bands − RGB | -0.014 [-0.034, +0.006] |
| Logistic regression: 204 bands − RGB | +0.371 [+0.341, +0.404] |
| 204 bands: CNN − logistic regression | +0.040 [+0.023, +0.058] |
| RGB: CNN − logistic regression | +0.425 [+0.395, +0.457] |

## CNN detail

| input | seeds | seed mean | R1 | R2 | R3 | R4 | R5 | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `all` | 0.889, 0.883, 0.854 | 0.875 | 0.670 | 0.960 | 0.995 | 0.995 | 0.950 | 13 |
| `rgb` | 0.916, 0.931, 0.896 | 0.914 | 0.845 | 0.970 | 0.995 | 0.965 | 0.865 | 70 |
