# Red blotch from leaf spectra, within Chardonnay

Can a spectrum-based model detect red blotch inside one variety (Chardonnay, 250 leaves: 88 red blotch,
162 not)? Every leaf becomes one row of features. The models are classical: logistic regression,
PLS-DA and shallow gradient boosting. Scoring and folds are the same as the whole-leaf CNN in
`../../clean/disease_classification/attention_mil/`, so the numbers can sit side by side.

```bash
cd "claude runs/red_blotch_spectra"
../../.venv/bin/python -u red_blotch_spectra.py                      # all pipelines × 3 fold schemes + 1000 permutations, ~5 min
../../.venv/bin/python -u red_blotch_spectra.py --n-perm 20          # quicker check of the whole flow, ~40 s
```

## Pre-declared analysis (written before any result existed)

- **Primary comparison:** feature set **D** (each vegetation index's spread across the leaf) with
  logistic regression, against the baseline (SNV leaf-mean spectrum + logistic regression, C = 1).
  It is scored with `block` folds, by within-row AUC. A p-value comes from a permutation test that reruns the
  whole pipeline (inner CV included) with labels shuffled within each row.
  - **Hypothesis:** red blotch shows as patchy changes, so how an index varies over the leaf carries
    signal that the leaf mean washes out.
- **Everything else is exploratory:** the other 13 pipelines, and the `row` and `photo` schemes.
  - **Best pipeline:** the best linear pipeline gets a family-wise p-value from a max-statistic null (the best of
    all linear pipelines in each permutation). Gradient boosting is too slow to permute and is reported
    without a p-value.
  - **Leaf size:** the leaf's pixel count is scored on its own, so a gain from D can be checked against
    "it just measures leaf size".
- **Added after seeing results:** one more pipeline, "best single feature", was added after the first run
  showed single indices doing well on their own. It picks one feature from E on the training folds only.
  It's post hoc and exploratory, but it's included in the family-wise permutation null.

## Features (one row per leaf)

| set | what | count |
|---|---|---|
| A | SNV leaf-mean spectrum (all leaf pixels, from `../shared/cache`) | 204 |
| B | first derivative (Savitzky–Golay, window 11, order 2) of the SNV leaf-mean spectrum | 204 |
| C | 11 vegetation indices of the leaf-mean spectrum (NDVI, NDRE, CI red-edge, red-edge position, PRI, ARI, CRI, MCARI, SIPI, water index, green/red) | 11 |
| D | each index per pixel, then its 10th, 50th, 90th percentile and IQR over the leaf's **interior** pixels | 44 |
| E | C + D | 55 |

Interior pixels for D: the leaf mask eroded by 2 px (edge pixels mix leaf and background, and a small
leaf has relatively more of them, which would turn p10 into a leaf-size measure). Glint pixels (any
band > 1.0) are dropped too. Reflectance is clamped at 0.001 before the indices that divide by it.

## Models and evaluation

- **Models:**
  - best single feature from E, with its sign, picked on the training folds by within-row AUC;
  - logistic regression, L2, class-balanced; C from {0.01, 0.1, 1, 10};
  - PLS-DA, components from {1, 2, 3, 5, 8, 12};
  - gradient boosting (C, D, E only), depth {2, 3} × {50, 150} trees.

  StandardScaler first. Settings are picked by an inner CV over the 4 training folds of the same scheme,
  scored by within-row AUC.
- **Fold schemes:** the same as the CNN's (identical fold assignment, verified):
  - `row`: unseen rows;
  - `photo`: random photos dealt evenly per row, 2 per fold (R1 has 13 photos, so 3 or 2);
  - `block`: fold k holds the photos of each row whose first vine is in 10k+1 to 10k+10.

  A photo is the camera image a leaf was cut from, from `../../clean/leaf_sources.csv`, and never splits across folds.

  Unlike the CNN, these models need no validation fold, so they train on 4 folds; the CNN trained on 3.
- **Within-row AUC:** compares leaves of the same fold and the same row only. Each score comes with a 95%
  bootstrap CI (2000 resamples within fold × row × class) and a paired Δ against the baseline.

| output | what it is |
|---|---|
| `results/metrics.md` / `.json` | every pipeline × scheme: within-row AUC, CI, Δ vs baseline; permutation p-values; leaf-size check |
| `results/oof_<scheme>.csv` | per leaf: fold, label, score of every pipeline |
| `results/univariate.csv` | each C/D feature and leaf size alone: within-row AUC with CI (no model) |
| `cache/leaf_features.npz` | the feature table (rebuilt if missing) |
| `figures/auc_by_pipeline.png` | the main figure |
| `figures/feature_auc.png` | which single features separate red blotch best |
| `logs/run.log` | what the last run printed |

Results and what they mean: `REPORT.md`.
