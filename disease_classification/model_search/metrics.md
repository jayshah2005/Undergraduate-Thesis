# Disease model search: all scores

Secondary scores and diagnostics behind `final_report.md`. Test rows unless marked validation.

## Leafroll

Secondary search scores (within: mean of the 3 varieties):

| pool | within pick | own-row pick | top-5 | top-10 | within-row AUC of the pick | pooled pick | own-row pick | top-5 | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| untuned families | 0.514 | 0.492 | 0.517 | 0.532 | 0.541 | 0.657 | 0.619 | 0.659 | 0.679 |
| tuned | 0.514 | 0.494 | 0.523 | 0.515 | 0.515 | 0.661 | 0.630 | 0.649 | 0.662 |
| literature | 0.517 | 0.510 | 0.518 | 0.529 | 0.477 | 0.654 | 0.681 | 0.672 | 0.667 |
| every candidate | 0.508 | 0.492 | 0.530 | 0.513 | 0.528 | 0.644 | 0.593 | 0.656 | 0.662 |

*Own-row pick*: test row k uses fold k's own validation row only. *Top-k*: mean of the k best on validation. *Within-row AUC*: AUC inside each variety × row, then averaged; rows differ in infection rate, so a model can score above 0.5 across rows with no signal inside any row.

Picks, test rows R1–R5:

- untuned families, within: `lda:mt_rownorm_snv_vis@pooled`, `tabpfn:texture_idx@within`, `svm:capnorm_snv_vis@pooled`, `svm:dino_small_cir_mean@pooled`, `tabpfn:texture_idx@within`
- untuned families, pooled: `pca_lr:dino_small_rgb_both@within`, `lr:snvq_spread@pooled`, `svm:dino_small_rgb_cls@pooled`, `sil_logit:pix5_510_710@within`, `lr:texture@pooled`
- tuned, within: `tuned:tabpfn:idx_mean@within`, `tuned:tabpfn:texture_idx@within`, `tuned:tabpfn:texture_idx@within`, `tuned:pls:rownorm_snv_vis@within`, `tuned:tabpfn:texture_idx@within`
- tuned, pooled: `tuned:svm:idx@pooled`, `tuned:lr:idx@within`, `tuned:svm:idx_mean@within`, `tuned:lr:snv_mean@pooled`, `tuned:svm:idx_mean@within`
- literature, within: `lit:svm:prospect_pig@pooled`, `lit:lr:prospect_pig@within`, `lit:svm:prospect_pig@pooled`, `lit:lr:prospect@within`, `lit:svm:prospect_pig@pooled`
- literature, pooled: `lit:lr:prospect_pig@pooled`, `lit:lr:prospect_pig@pooled`, `lit:lr:prospect_pig@pooled`, `lit:lr:prospect_pig@pooled`, `lit:lr:prospect_pig@pooled`
- every candidate, within: `tuned:tabpfn:idx_mean@within`, `tuned:tabpfn:texture_idx@within`, `tuned:tabpfn:texture_idx@within`, `svm:dino_small_cir_mean@pooled`, `tuned:tabpfn:texture_idx@within`
- every candidate, pooled: `pca_lr:dino_small_rgb_both@within`, `lr:snvq_spread@pooled`, `tuned:svm:idx_mean@within`, `sil_logit:pix5_510_710@within`, `lr:texture@pooled`

Over all 515 candidates: test balanced accuracy median 0.502, 95th percentile 0.540, best 0.584 (chosen after seeing test, so not a result); within-row AUC median 0.504, 95th percentile 0.554.

Pooled ceiling on validation (each candidate's best per-variety thresholds, chosen on the same labels): median 0.683, best 0.714.

## Red blotch

Secondary search scores (within: mean of the 3 varieties):

| pool | within pick | own-row pick | top-5 | top-10 | within-row AUC of the pick | pooled pick | own-row pick | top-5 | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| untuned families | 0.539 | 0.552 | 0.570 | 0.595 | 0.555 | 0.637 | 0.595 | 0.639 | 0.633 |
| tuned | 0.534 | 0.574 | 0.551 | 0.551 | 0.554 | 0.620 | 0.590 | 0.628 | 0.623 |
| literature | 0.571 | 0.543 | 0.583 | 0.590 | 0.579 | 0.612 | 0.613 | 0.636 | 0.639 |
| every candidate | 0.543 | 0.562 | 0.575 | 0.558 | 0.553 | 0.637 | 0.586 | 0.637 | 0.632 |

*Own-row pick*: test row k uses fold k's own validation row only. *Top-k*: mean of the k best on validation. *Within-row AUC*: AUC inside each variety × row, then averaged; rows differ in infection rate, so a model can score above 0.5 across rows with no signal inside any row.

Picks, test rows R1–R5:

- untuned families, within: `hgb:idx@within`, `svm:mt_rownorm_rgb@pooled`, `svm:snv_vis@within`, `svm:snv_vis@within`, `lda:rownorm_rgb@within`
- untuned families, pooled: `svm:rownorm_rgb@within`, `svm:rownorm_rgb@within`, `svm:rownorm_rgb@within`, `lda:refl_vis@within`, `svm:rownorm_rgb@within`
- tuned, within: `tuned:lda:idx_mean@within`, `tuned:lda:idx_mean@within`, `tuned:lda:d1@within`, `tuned:pls:refl_vis@within`, `tuned:svm:rownorm_snv_vis@within`
- tuned, pooled: `tuned:knn:snv_vis@pooled`, `tuned:lr:refl_vis@within`, `tuned:tabpfn:snv_vis@within`, `tuned:knn:idx_mean@pooled`, `tuned:knn:idx_mean@pooled`
- literature, within: `lit:lr:prospect_pig@pooled`, `lit:lda:prospect_pig@within`, `lit:lda:prospect_pig@within`, `lit:lr:prospect_pig@pooled`, `lit:svm:prospect@within`
- literature, pooled: `lit:lda:prospect@pooled`, `lit:lda:prospect@within`, `lit:joint4_lr:snv_vis@within`, `lit:lr:prospect@within`, `lit:rf:prospect_pig@within`
- every candidate, within: `lit:lr:prospect_pig@pooled`, `svm:mt_rownorm_rgb@pooled`, `svm:snv_vis@within`, `svm:snv_vis@within`, `lda:rownorm_rgb@within`
- every candidate, pooled: `svm:rownorm_rgb@within`, `svm:rownorm_rgb@within`, `svm:rownorm_rgb@within`, `lda:refl_vis@within`, `svm:rownorm_rgb@within`

Over all 515 candidates: test balanced accuracy median 0.538, 95th percentile 0.579, best 0.599 (chosen after seeing test, so not a result); within-row AUC median 0.554, 95th percentile 0.601.

Pooled ceiling on validation (each candidate's best per-variety thresholds, chosen on the same labels): median 0.632, best 0.664.

## Tuning

Tuned minus untuned, same algorithm, features and mode (mean balanced accuracy over the 3 varieties; the test column was computed after the tuned headline was scored):

| algorithm | pairs | validation leafroll | validation red blotch | test leafroll | test red blotch |
|---|---|---|---|---|---|
| cnn | 6 | +0.003 | -0.018 | -0.008 | -0.016 |
| hgb | 4 | -0.001 | -0.002 | +0.015 | +0.004 |
| lda | 16 | -0.004 | +0.006 | -0.015 | +0.005 |
| lr | 16 | -0.005 | -0.012 | +0.011 | -0.006 |
| rf | 4 | +0.007 | +0.002 | -0.015 | -0.000 |
| svm | 16 | +0.009 | -0.009 | +0.008 | -0.006 |
| tabpfn | 4 | +0.036 | -0.027 | +0.021 | -0.024 |

## Split comparison (Sawyer et al. 2023)

The same models with Sawyer et al.'s split (stratified 5-fold over shuffled leaves, 5 repeats) and with whole rows held out (leave one row out, so 4 training rows). Accuracy / balanced accuracy. Not a result: shuffled folds share rows between training and test.

| leaves | task | model | shuffled | rows held out | majority-class accuracy |
|---|---|---|---|---|---|
| Baco | Leafroll | RF, 510-710 nm (Sawyer) | 0.512 / 0.505 | 0.468 / 0.460 | 0.569 |
| Baco | Leafroll | LR, SNV mean (repo baseline) | 0.559 / 0.556 | 0.484 / 0.481 | 0.569 |
| Baco | Red blotch | RF, 510-710 nm (Sawyer) | 0.549 / 0.547 | 0.532 / 0.536 | 0.544 |
| Baco | Red blotch | LR, SNV mean (repo baseline) | 0.513 / 0.510 | 0.492 / 0.490 | 0.544 |
| Chardonnay | Leafroll | RF, 510-710 nm (Sawyer) | 0.521 / 0.520 | 0.540 / 0.539 | 0.512 |
| Chardonnay | Leafroll | LR, SNV mean (repo baseline) | 0.616 / 0.616 | 0.504 / 0.503 | 0.512 |
| Chardonnay | Red blotch | RF, 510-710 nm (Sawyer) | 0.607 / 0.565 | 0.584 / 0.557 | 0.648 |
| Chardonnay | Red blotch | LR, SNV mean (repo baseline) | 0.629 / 0.611 | 0.600 / 0.582 | 0.648 |
| Vidal | Leafroll | RF, 510-710 nm (Sawyer) | 0.810 / 0.510 | 0.808 / 0.556 | 0.900 |
| Vidal | Leafroll | LR, SNV mean (repo baseline) | 0.759 / 0.578 | 0.752 / 0.524 | 0.900 |
| Vidal | Red blotch | RF, 510-710 nm (Sawyer) | 0.559 / 0.512 | 0.552 / 0.511 | 0.676 |
| Vidal | Red blotch | LR, SNV mean (repo baseline) | 0.594 / 0.572 | 0.552 / 0.537 | 0.676 |
| Baco + Chardonnay + Vidal | Leafroll | RF, 510-710 nm (Sawyer) | 0.606 / 0.579 | 0.599 / 0.571 | 0.660 |
| Baco + Chardonnay + Vidal | Leafroll | LR, SNV mean (repo baseline) | 0.627 / 0.632 | 0.590 / 0.578 | 0.660 |
| Baco + Chardonnay + Vidal | Red blotch | RF, 510-710 nm (Sawyer) | 0.599 / 0.598 | 0.590 / 0.590 | 0.524 |
| Baco + Chardonnay + Vidal | Red blotch | LR, SNV mean (repo baseline) | 0.586 / 0.586 | 0.592 / 0.593 | 0.524 |

## Permutation nulls

Labels shuffled within each variety × row (keeps every row's infection rate), the whole candidate rerun, inner tuning included. Validation, mean of the 3 varieties.

| candidate | task | observed | null mean | null 95th pct | shuffles |
|---|---|---|---|---|---|
| `lda:rownorm_snv_vis@within` | Leafroll | 0.554 | 0.496 | 0.534 | 50 |
| `lda:rownorm_snv_vis@within` | Red blotch | 0.578 | 0.507 | 0.543 | 50 |
| `svm:snv_vis@within` | Leafroll | 0.508 | 0.493 | 0.527 | 50 |
| `svm:snv_vis@within` | Red blotch | 0.589 | 0.512 | 0.541 | 50 |

## PROSPECT-D traits

Within-row standardised difference (diseased − healthy) of each fitted trait, averaged over rows (all 5 rows, computed after the literature headline was scored). Pigments are relative PROSPECT-fitted values.

| task | variety | N | Cab | Car | Ant | Cbrown | Cm | Car/Cab | Ant/Cab |
|---|---|---|---|---|---|---|---|---|---|
| Leafroll | Baco | +0.12 | -0.07 | -0.02 | -0.12 | +0.19 | +0.01 | +0.33 | -0.11 |
| Leafroll | Chardonnay | -0.06 | -0.05 | -0.06 | -0.06 | +0.22 | -0.12 | +0.05 | +0.05 |
| Leafroll | Vidal | -0.16 | -0.07 | -0.19 | -0.25 | -0.33 | +0.12 | -0.75 | -0.17 |
| Red blotch | Baco | -0.18 | +0.08 | +0.04 | -0.09 | -0.04 | -0.22 | -0.23 | -0.13 |
| Red blotch | Chardonnay | -0.47 | -0.57 | -0.55 | -0.33 | +0.03 | -0.46 | +0.37 | +0.20 |
| Red blotch | Vidal | -0.27 | -0.35 | -0.33 | -0.41 | +0.11 | -0.13 | +0.23 | +0.03 |
