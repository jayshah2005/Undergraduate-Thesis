# Variety CNN: metrics

Balanced accuracy, chance 0.25; held-out field rows; 95% bootstrap interval. Every model uses SNV; CNN scores are 3-seed ensembles. `_no_val_e<N>` = trained on 4 rows for exactly N epochs, with no validation row. `_nested` = the same, with N chosen inside each training split.

| model | balanced accuracy |
|---|---|
| CNN `all` | 0.898 (0.880–0.916) |
| CNN `all_no_val_e37` | 0.949 (0.936–0.961) |
| CNN `all_nested` | 0.925 (0.909–0.940) |
| CNN `rgb` | 0.824 (0.799–0.846) |
| CNN `rgb_no_val_e37` | 0.838 (0.816–0.860) |
| CNN `rgb_nested` | 0.876 (0.857–0.896) |
| logistic regression `all`, same 3 rows | 0.849 (0.827–0.871) |
| logistic regression `rgb`, same 3 rows | 0.488 (0.457–0.517) |
| logistic regression `all`, 4 rows | 0.939 (0.924–0.952) |
| logistic regression `rgb`, 4 rows | 0.500 (0.468–0.530) |

Leaf size alone (pixel count, height, width), logistic regression: 0.654.

## Paired differences (same resampled leaves)

| comparison | Δ balanced accuracy [95% CI] |
|---|---|
| CNN: 204 bands − RGB | +0.074 [+0.049, +0.100] |
| Logistic regression (3 rows): 204 bands − RGB | +0.361 [+0.328, +0.394] |
| 204 bands: CNN − logistic regression (3 rows) | +0.049 [+0.031, +0.068] |
| RGB: CNN − logistic regression (3 rows) | +0.336 [+0.302, +0.371] |
| CNN without validation, 37 epochs each: 204 bands − RGB | +0.111 [+0.088, +0.134] |
| CNN, epochs chosen inside each split: 204 bands − RGB | +0.049 [+0.027, +0.072] |
| 204 bands: CNN, epochs chosen inside each split − logistic regression (4 rows) | -0.014 [-0.032, +0.004] |
| RGB: CNN, epochs chosen inside each split − logistic regression (4 rows) | +0.376 [+0.344, +0.411] |
| Logistic regression (4 rows): 204 bands − RGB | +0.439 [+0.408, +0.470] |
| 204 bands: CNN, 37 epochs, no validation − logistic regression (4 rows) | +0.010 [-0.005, +0.026] |
| RGB: CNN, 37 epochs, no validation − logistic regression (4 rows) | +0.338 [+0.305, +0.372] |

## Epochs chosen inside each training split

Average of the 5 row scores for each of the 3 single models, and the epoch count each test row got.

| input | row average per seed | mean | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|---|
| `all` | 0.866, 0.806, 0.771 | 0.814 | 67 | 34 | 76 | 64 | 69 |
| `rgb` | 0.853, 0.884, 0.880 | 0.872 | 64 | 70 | 77 | 80 | 57 |

## CNN detail

| configuration | seeds | R1 | R2 | R3 | R4 | R5 | median best epoch |
|---|---|---|---|---|---|---|---|
| `all` | 0.857, 0.856, 0.914 | 0.595 | 0.975 | 0.975 | 0.975 | 0.970 | 11 |
| `all_no_val_e37` | 0.826, 0.932, 0.744 | 0.915 | 0.970 | 0.964 | 0.960 | 0.935 | 37 |
| `all_nested` | 0.866, 0.805, 0.772 | 0.845 | 0.950 | 0.975 | 0.940 | 0.915 | 67 |
| `rgb` | 0.780, 0.791, 0.788 | 0.730 | 0.870 | 0.905 | 0.820 | 0.795 | 49 |
| `rgb_no_val_e37` | 0.814, 0.844, 0.857 | 0.880 | 0.875 | 0.855 | 0.825 | 0.755 | 37 |
| `rgb_nested` | 0.853, 0.884, 0.880 | 0.885 | 0.865 | 0.915 | 0.870 | 0.845 | 70 |
