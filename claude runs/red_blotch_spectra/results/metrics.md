# Red blotch from leaf spectra, Chardonnay: results

Within-row AUC: leaves of the same fold and row only (0.5 = chance), 95% bootstrap CI. Δ: paired difference to the baseline (A · logistic, C = 1) on the same resamples.

## Pre-declared primary: D index spread (pixels) · logistic vs baseline, `block` folds

- within-row AUC 0.595 [0.520, 0.669]; Δ vs baseline +0.018 [-0.077, +0.113]
- permutation p (labels shuffled within rows, whole pipeline rerun, 1000 permutations): **0.048** (null 95th percentile 0.592); baseline p = 0.075
- exploratory: best linear pipeline under `block` = D index spread (pixels) · logistic (AUC 0.595), family-wise p = 0.244 (max-statistic null 95th percentile 0.634)

## Leaf size alone (pixel count as the score)

| scheme | within-row AUC |
|---|---|
| `row` | 0.493 |
| `photo` | 0.505 |
| `block` | 0.516 |

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

## `photo` folds: random photos dealt evenly per row, 2 per fold; R1 has 13 photos, so 3 or 2 (new vines in seen rows)

| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |
|---|---|---|---|---|---|
| A mean SNV spectrum · logistic, C=1 (baseline) | 0.604 [0.530, 0.677] | — | — | 0.654 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| A mean SNV spectrum · logistic | 0.593 [0.522, 0.661] | -0.010 [-0.087, +0.061] | 0.38 | 0.648 | 0.01, 0.01, 10.0, 0.01, 0.01 |
| A mean SNV spectrum · PLS-DA | 0.573 [0.504, 0.642] | -0.030 [-0.108, +0.049] | 0.20 | 0.620 | 5, 1, 3, 1, 1 |
| B mean 1st derivative · logistic | 0.563 [0.488, 0.634] | -0.041 [-0.102, +0.022] | 0.10 | 0.630 | 0.01, 0.01, 1.0, 0.01, 1.0 |
| B mean 1st derivative · PLS-DA | 0.575 [0.506, 0.646] | -0.028 [-0.104, +0.047] | 0.21 | 0.634 | 12, 1, 3, 1, 12 |
| C indices (leaf mean) · logistic | 0.622 [0.549, 0.693] | +0.018 [-0.063, +0.102] | 0.64 | 0.667 | 0.01, 0.01, 0.01, 0.01, 0.01 |
| C indices (leaf mean) · PLS-DA | 0.596 [0.526, 0.665] | -0.008 [-0.085, +0.071] | 0.40 | 0.655 | 1, 5, 2, 1, 1 |
| C indices (leaf mean) · gradient boosting | 0.607 [0.533, 0.684] | +0.003 [-0.089, +0.099] | 0.50 | 0.627 | (3, 150), (2, 150), (3, 50), (3, 150), (2, 50) |
| D index spread (pixels) · logistic | 0.598 [0.522, 0.671] | -0.006 [-0.100, +0.089] | 0.43 | 0.597 | 10.0, 0.01, 10.0, 0.01, 0.01 |
| D index spread (pixels) · PLS-DA | 0.602 [0.522, 0.679] | -0.002 [-0.089, +0.087] | 0.47 | 0.630 | 5, 1, 12, 1, 12 |
| D index spread (pixels) · gradient boosting | 0.530 [0.457, 0.604] | -0.073 [-0.157, +0.008] | 0.04 | 0.570 | (2, 50), (2, 150), (2, 150), (2, 150), (3, 150) |
| E = C + D · logistic | 0.589 [0.516, 0.665] | -0.014 [-0.106, +0.075] | 0.38 | 0.640 | 10.0, 10.0, 1.0, 0.01, 10.0 |
| E = C + D · PLS-DA | 0.585 [0.508, 0.665] | -0.018 [-0.112, +0.075] | 0.34 | 0.618 | 12, 1, 5, 1, 12 |
| E = C + D · gradient boosting | 0.563 [0.490, 0.638] | -0.041 [-0.122, +0.041] | 0.16 | 0.581 | (3, 50), (3, 50), (2, 150), (3, 150), (2, 50) |
| E = C + D · best single feature | 0.561 [0.484, 0.638] | -0.043 [-0.126, +0.041] | 0.15 | 0.557 | +GR p50, +MCARI (leaf mean), +GR p50, +GR (leaf mean), +GR p10 |

## `block` folds: fold k = the photos of each row whose first vine is in 10k+1 to 10k+10 (held-out vines are neighbours of each other)

| pipeline | within-row AUC [95% CI] | Δ vs baseline [95% CI] | P(Δ > 0) | pooled AUC | chosen setting per fold |
|---|---|---|---|---|---|
| A mean SNV spectrum · logistic, C=1 (baseline) | 0.577 [0.500, 0.653] | — | — | 0.660 | 1.0, 1.0, 1.0, 1.0, 1.0 |
| A mean SNV spectrum · logistic | 0.556 [0.477, 0.633] | -0.020 [-0.083, +0.043] | 0.25 | 0.586 | 10.0, 10.0, 0.01, 10.0, 10.0 |
| A mean SNV spectrum · PLS-DA | 0.586 [0.513, 0.662] | +0.009 [-0.052, +0.072] | 0.61 | 0.627 | 12, 12, 1, 12, 12 |
| B mean 1st derivative · logistic | 0.543 [0.464, 0.622] | -0.034 [-0.097, +0.032] | 0.14 | 0.555 | 10.0, 10.0, 0.01, 10.0, 1.0 |
| B mean 1st derivative · PLS-DA | 0.559 [0.480, 0.642] | -0.018 [-0.086, +0.052] | 0.29 | 0.570 | 12, 12, 1, 3, 12 |
| C indices (leaf mean) · logistic | 0.565 [0.489, 0.644] | -0.011 [-0.099, +0.081] | 0.38 | 0.652 | 10.0, 0.01, 0.01, 0.01, 0.01 |
| C indices (leaf mean) · PLS-DA | 0.590 [0.511, 0.669] | +0.014 [-0.079, +0.104] | 0.60 | 0.628 | 1, 1, 1, 1, 1 |
| C indices (leaf mean) · gradient boosting | 0.566 [0.486, 0.649] | -0.010 [-0.101, +0.090] | 0.40 | 0.604 | (2, 150), (2, 150), (2, 150), (3, 50), (3, 50) |
| D index spread (pixels) · logistic | 0.595 [0.520, 0.669] | +0.018 [-0.077, +0.113] | 0.63 | 0.631 | 1.0, 0.1, 0.01, 0.01, 10.0 |
| D index spread (pixels) · PLS-DA | 0.556 [0.477, 0.633] | -0.020 [-0.117, +0.074] | 0.32 | 0.575 | 5, 8, 2, 5, 8 |
| D index spread (pixels) · gradient boosting | 0.437 [0.358, 0.518] | -0.140 [-0.232, -0.045] | 0.00 | 0.524 | (2, 50), (2, 50), (3, 50), (2, 50), (3, 50) |
| E = C + D · logistic | 0.588 [0.511, 0.660] | +0.011 [-0.090, +0.106] | 0.56 | 0.640 | 1.0, 10.0, 0.01, 10.0, 10.0 |
| E = C + D · PLS-DA | 0.541 [0.462, 0.619] | -0.036 [-0.133, +0.061] | 0.21 | 0.580 | 8, 12, 2, 5, 8 |
| E = C + D · gradient boosting | 0.498 [0.421, 0.583] | -0.079 [-0.178, +0.020] | 0.06 | 0.508 | (2, 150), (2, 150), (3, 50), (3, 150), (2, 50) |
| E = C + D · best single feature | 0.568 [0.491, 0.644] | -0.009 [-0.104, +0.086] | 0.41 | 0.574 | +GR p50, +CRI p50, +MCARI (leaf mean), +GR p50, +MCARI (leaf mean) |

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
