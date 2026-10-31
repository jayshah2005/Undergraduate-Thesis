# Red blotch from leaf spectra, Chardonnay: results

Within-row AUC: leaves of the same fold and row only (0.5 = chance), 95% bootstrap CI. Δ: paired difference to the baseline (A · logistic, C = 1) on the same resamples.

## Pre-declared primary: D index spread (pixels) · logistic vs baseline, `block` folds

- within-row AUC 0.558 [0.482, 0.634]; Δ vs baseline -0.027 [-0.125, +0.071]
- permutation p (labels shuffled within rows, whole pipeline rerun, 1000 permutations): **0.157** (null 95th percentile 0.594); baseline p = 0.059
- exploratory: best linear pipeline under `block` = A mean SNV spectrum · PLS-DA (AUC 0.609), family-wise p = 0.135 (max-statistic null 95th percentile 0.633)

## Leaf size alone (pixel count as the score)

| scheme | within-row AUC |
|---|---|
| `row` | 0.493 |
| `photo` | 0.494 |
| `block` | 0.522 |

## `row` folds: fold k = field row k (unseen rows)

| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |
|---|---|---|---|---|---|
| A mean SNV spectrum · logistic, C=1 (baseline) | 0.565 [0.483, 0.647] | — | — | 0.612 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| A mean SNV spectrum · logistic | 0.554 [0.470, 0.633] | -0.011 [-0.084, +0.061] | 0.35 | 0.588 | 0.01, 0.01, 10.0, 0.01, 0.01 |
| A mean SNV spectrum · PLS-DA | 0.574 [0.493, 0.651] | +0.009 [-0.069, +0.088] | 0.57 | 0.577 | 1, 1, 12, 1, 1 |
| B mean 1st derivative · logistic | 0.549 [0.468, 0.631] | -0.016 [-0.082, +0.056] | 0.32 | 0.576 | 0.01, 0.01, 1.0, 0.01, 0.01 |
| B mean 1st derivative · PLS-DA | 0.556 [0.474, 0.638] | -0.009 [-0.087, +0.071] | 0.40 | 0.561 | 1, 1, 12, 1, 1 |
| C indices (leaf mean) · logistic | 0.622 [0.541, 0.700] | +0.058 [-0.025, +0.138] | 0.92 | 0.637 | 0.01, 0.01, 0.01, 0.01, 0.01 |
| C indices (leaf mean) · PLS-DA | 0.634 [0.555, 0.708] | +0.069 [-0.011, +0.153] | 0.95 | 0.639 | 1, 1, 1, 1, 1 |
| C indices (leaf mean) · gradient boosting | 0.612 [0.532, 0.683] | +0.047 [-0.045, +0.138] | 0.84 | 0.623 | (2, 50), (3, 50), (2, 150), (2, 150), (2, 150) |
| D index spread (pixels) · logistic | 0.625 [0.547, 0.700] | +0.060 [-0.025, +0.142] | 0.92 | 0.604 | 10.0, 0.1, 10.0, 10.0, 10.0 |
| D index spread (pixels) · PLS-DA | 0.635 [0.555, 0.710] | +0.070 [-0.013, +0.151] | 0.94 | 0.593 | 12, 1, 1, 12, 12 |
| D index spread (pixels) · gradient boosting | 0.523 [0.439, 0.603] | -0.042 [-0.131, +0.043] | 0.16 | 0.503 | (3, 150), (2, 50), (3, 50), (2, 150), (2, 150) |
| E = C + D · logistic | 0.589 [0.508, 0.665] | +0.024 [-0.061, +0.111] | 0.72 | 0.558 | 10.0, 0.01, 10.0, 0.1, 10.0 |
| E = C + D · PLS-DA | 0.627 [0.545, 0.702] | +0.062 [-0.022, +0.147] | 0.93 | 0.634 | 1, 1, 1, 1, 1 |
| E = C + D · gradient boosting | 0.521 [0.439, 0.600] | -0.044 [-0.130, +0.040] | 0.14 | 0.548 | (2, 50), (2, 50), (3, 150), (3, 50), (2, 150) |
| E = C + D · best single feature | 0.620 [0.543, 0.692] | +0.055 [-0.027, +0.135] | 0.90 | 0.586 | +MCARI (leaf mean), +GR p50, +MCARI (leaf mean), −NDRE (leaf mean), +MCARI (leaf mean) |

## `photo` folds: fold k = 2 random photos from every row (new vines in seen rows)

| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |
|---|---|---|---|---|---|
| A mean SNV spectrum · logistic, C=1 (baseline) | 0.627 [0.545, 0.703] | — | — | 0.674 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| A mean SNV spectrum · logistic | 0.621 [0.541, 0.698] | -0.006 [-0.069, +0.058] | 0.40 | 0.632 | 10.0, 10.0, 10.0, 0.01, 10.0 |
| A mean SNV spectrum · PLS-DA | 0.612 [0.539, 0.683] | -0.015 [-0.084, +0.056] | 0.32 | 0.638 | 5, 12, 12, 1, 12 |
| B mean 1st derivative · logistic | 0.599 [0.524, 0.677] | -0.028 [-0.088, +0.034] | 0.16 | 0.653 | 10.0, 1.0, 10.0, 1.0, 0.1 |
| B mean 1st derivative · PLS-DA | 0.582 [0.502, 0.659] | -0.045 [-0.108, +0.017] | 0.08 | 0.643 | 12, 12, 1, 12, 5 |
| C indices (leaf mean) · logistic | 0.621 [0.550, 0.700] | -0.006 [-0.082, +0.071] | 0.44 | 0.658 | 0.01, 0.01, 10.0, 0.01, 0.01 |
| C indices (leaf mean) · PLS-DA | 0.610 [0.539, 0.683] | -0.017 [-0.091, +0.060] | 0.33 | 0.639 | 1, 1, 1, 1, 2 |
| C indices (leaf mean) · gradient boosting | 0.631 [0.556, 0.707] | +0.004 [-0.088, +0.093] | 0.55 | 0.643 | (3, 50), (2, 50), (2, 150), (3, 150), (3, 50) |
| D index spread (pixels) · logistic | 0.603 [0.522, 0.675] | -0.024 [-0.116, +0.067] | 0.30 | 0.608 | 10.0, 0.01, 10.0, 10.0, 0.1 |
| D index spread (pixels) · PLS-DA | 0.595 [0.519, 0.668] | -0.032 [-0.119, +0.052] | 0.23 | 0.611 | 8, 1, 5, 1, 3 |
| D index spread (pixels) · gradient boosting | 0.448 [0.373, 0.528] | -0.179 [-0.269, -0.086] | 0.00 | 0.533 | (3, 150), (3, 50), (2, 50), (2, 50), (2, 50) |
| E = C + D · logistic | 0.584 [0.506, 0.657] | -0.043 [-0.134, +0.045] | 0.16 | 0.633 | 10.0, 0.01, 10.0, 0.01, 0.01 |
| E = C + D · PLS-DA | 0.606 [0.539, 0.677] | -0.022 [-0.101, +0.062] | 0.31 | 0.634 | 5, 1, 5, 1, 3 |
| E = C + D · gradient boosting | 0.517 [0.442, 0.597] | -0.110 [-0.198, -0.019] | 0.01 | 0.549 | (2, 50), (3, 150), (2, 50), (2, 50), (2, 150) |
| E = C + D · best single feature | 0.631 [0.554, 0.709] | +0.004 [-0.080, +0.091] | 0.53 | 0.588 | +MCARI (leaf mean), +GR p50, +MCARI (leaf mean), +MCARI (leaf mean), +GR p50 |

## `block` folds: fold k = vines 10k+1 to 10k+10 of every row (held-out vines are neighbours of each other)

| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |
|---|---|---|---|---|---|
| A mean SNV spectrum · logistic, C=1 (baseline) | 0.585 [0.509, 0.661] | — | — | 0.681 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| A mean SNV spectrum · logistic | 0.547 [0.471, 0.618] | -0.038 [-0.098, +0.022] | 0.10 | 0.585 | 10.0, 10.0, 0.01, 0.01, 10.0 |
| A mean SNV spectrum · PLS-DA | 0.609 [0.538, 0.688] | +0.025 [-0.029, +0.078] | 0.81 | 0.655 | 12, 12, 3, 12, 12 |
| B mean 1st derivative · logistic | 0.571 [0.496, 0.647] | -0.013 [-0.076, +0.051] | 0.35 | 0.610 | 10.0, 1.0, 0.01, 10.0, 1.0 |
| B mean 1st derivative · PLS-DA | 0.583 [0.504, 0.658] | -0.002 [-0.065, +0.067] | 0.46 | 0.614 | 12, 12, 3, 3, 12 |
| C indices (leaf mean) · logistic | 0.554 [0.478, 0.625] | -0.031 [-0.118, +0.056] | 0.23 | 0.656 | 10.0, 0.01, 0.01, 0.01, 0.01 |
| C indices (leaf mean) · PLS-DA | 0.589 [0.513, 0.661] | +0.004 [-0.083, +0.092] | 0.52 | 0.638 | 1, 1, 1, 1, 1 |
| C indices (leaf mean) · gradient boosting | 0.567 [0.491, 0.645] | -0.018 [-0.112, +0.074] | 0.34 | 0.605 | (3, 150), (3, 150), (2, 150), (3, 50), (2, 150) |
| D index spread (pixels) · logistic | 0.558 [0.482, 0.634] | -0.027 [-0.125, +0.071] | 0.29 | 0.642 | 1.0, 10.0, 0.01, 0.01, 10.0 |
| D index spread (pixels) · PLS-DA | 0.578 [0.500, 0.658] | -0.007 [-0.105, +0.094] | 0.44 | 0.581 | 1, 8, 2, 3, 12 |
| D index spread (pixels) · gradient boosting | 0.453 [0.373, 0.531] | -0.132 [-0.230, -0.036] | 0.01 | 0.507 | (2, 150), (2, 50), (3, 50), (2, 50), (2, 50) |
| E = C + D · logistic | 0.594 [0.522, 0.663] | +0.009 [-0.087, +0.103] | 0.57 | 0.656 | 1.0, 10.0, 0.01, 10.0, 10.0 |
| E = C + D · PLS-DA | 0.556 [0.478, 0.630] | -0.029 [-0.118, +0.065] | 0.27 | 0.589 | 1, 8, 2, 2, 8 |
| E = C + D · gradient boosting | 0.504 [0.429, 0.578] | -0.080 [-0.167, +0.007] | 0.04 | 0.575 | (2, 50), (2, 150), (3, 50), (3, 50), (2, 50) |
| E = C + D · best single feature | 0.569 [0.495, 0.638] | -0.016 [-0.109, +0.076] | 0.37 | 0.513 | +MCARI (leaf mean), +CRI p50, +MCARI (leaf mean), +GR p50, +MCARI (leaf mean) |

## Single features (no model), within-row AUC, top 20

AUC > 0.5: higher on red-blotch leaves. 57 features screened, so expect ~3 with a CI excluding 0.5 by chance.

| feature | AUC [95% CI] |
|---|---|
| MCARI (leaf mean) | 0.661 [0.584, 0.735] |
| GR p50 | 0.657 [0.581, 0.727] |
| CIre (leaf mean) | 0.351 [0.282, 0.425] |
| NDRE (leaf mean) | 0.351 [0.282, 0.425] |
| REP (leaf mean) | 0.352 [0.283, 0.425] |
| GR (leaf mean) | 0.646 [0.573, 0.721] |
| GR p90 | 0.645 [0.570, 0.720] |
| MCARI p50 | 0.633 [0.557, 0.707] |
| GR p10 | 0.633 [0.559, 0.706] |
| REP p10 | 0.367 [0.295, 0.442] |
| CIre p10 | 0.367 [0.297, 0.445] |
| NDRE p10 | 0.367 [0.297, 0.445] |
| GR IQR | 0.621 [0.542, 0.700] |
| REP p50 | 0.379 [0.304, 0.454] |
| NDRE p50 | 0.390 [0.317, 0.468] |
| CIre p50 | 0.390 [0.317, 0.468] |
| MCARI p90 | 0.608 [0.529, 0.684] |
| CRI p50 | 0.608 [0.533, 0.681] |
| MCARI p10 | 0.597 [0.525, 0.667] |
| MCARI IQR | 0.597 [0.517, 0.672] |
