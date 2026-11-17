# What we tested

Scores are balanced accuracy unless marked AUC. Chance is 0.25 for variety and 0.5 for disease. Disease is scored per disease (leafroll, red blotch), never averaged. Items marked *removed* are in git history at the commit given.

## In the repo

### Variety (4 classes)

| Test | Outcome |
|---|---|
| Logistic regression on the SNV leaf-mean spectrum, 204 bands / RGB | 0.874 / 0.503 |
| Whole-leaf CNN (`small_cnn`), 204 bands / RGB | 0.898 / 0.824 |
| Small ResNet backbone (He et al. 2016), 204 bands / RGB | 0.907 / 0.885 |
| Attention pooling (Ilse et al. 2018) instead of the mean, small CNN and small ResNet, 204 bands / RGB | no gain anywhere (paired Δ −0.005 to +0.005); worse for the small ResNet on RGB (0.834 vs 0.885) |

Row R1 is the weak fold for every model (about 0.6).

### Disease

| Test | Outcome |
|---|---|
| Logistic regression baseline, per variety (Baco, Chardonnay, Vidal), 204 bands and RGB | 0.42–0.62; Chardonnay red blotch is the clearest (0.605) |
| CNN within variety, mean vs attention pooling (Ilse et al. 2018) | 0.37–0.59; no reliable gain over logistic regression |
| CNN pooled (all 998 leaves; Baco + Chardonnay + Vidal) | higher overall (red blotch 0.686) but near chance per variety; the gain is variety disease rates |
| 204 bands vs RGB | about equal |
| Small ResNet backbone on every disease run | in each run's `final_report.md` |
| Small ResNet without max pooling (stride-2 convs, Springenberg et al. 2015) and full ResNet-18 (He et al. 2016), within variety, 204 bands and RGB | no gain: within 0.06 of the small ResNet, ResNet-18 the weakest; the one interval above logistic regression (Baco leafroll) comes from calling whole rows positive or negative |
| Extra tasks: any disease vs healthy, both diseases | in the logistic regression reports; the pooled CNN (`all`) also runs any disease |

### Spectral analysis

| Test | Outcome |
|---|---|
| PCA of SNV leaf means by variety | variety dominates the spectra |
| PCA by disease class within variety (R² test) | only Chardonnay red blotch differs (R² 0.069, p 0.001, largest at 703 nm) |

## Removed from the repo (commit `fb310b9`)

| Test | Outcome |
|---|---|
| Backbone screen: HybridSN 3D CNN (Roy et al. 2020), 1D spectral CNN (Hu et al. 2015), ImageNet ResNet-18 with a band adaptor | lost to small ResNet on validation rows (the disease ranking is biased by early stopping on the same rows) |
| Variety CNN without rotations and flips, 204 bands / RGB | 0.914 / 0.928, against 0.898 / 0.824 with them |
| Disease model search, 515 candidates, aiming for 0.70 | none reached it; within variety 0.51–0.52 leafroll and 0.53–0.57 red blotch (logistic regression 0.535 / 0.554); pooled, nothing beat predicting from the variety's infection rate (0.679 / 0.615) |

What the disease search covered:
- Leaf features: SNV spectrum, RGB, 510–710 nm (Sawyer et al. 2023), first derivative, 18 vegetation indices, pixel quantiles, GLCM texture, leaf shape.
- Models: logistic regression, PCA + LR, SVM, LDA, PLS-DA, random forest, gradient boosting, k-NN, TabPFN v2, each also with a wide hyperparameter search.
- DINOv2 frozen embeddings of RGB and colour-infrared images.
- Per-row normalisation, per-photo centring, pixel-level logistic regression, multi-task per-variety weights, FiLM variety-conditioned CNN, CNNs with a clean validation row.
- From papers: PROSPECT-D pigment inversion, CARS-PLS-DA, SPA-LDA, joint 4-class co-infection model.
- Checks: permutation nulls; Sawyer et al.'s shuffled split, which scores only 0.50–0.63 here (they report 80–87%).

## Earlier work, before the rewrite (commit `387d886` and earlier)

| Test | Outcome |
|---|---|
| Transfer-learning study, 47 runs: classical models on leaf means and pixels, pixel MLP and 1D CNN, attention MIL, ResNet-18 (ImageNet, scratch, fixed RGB), DOFA, DINOv2, self-supervised pretraining, leave-one-variety-out | all tied; within-variety AUC 0.53–0.59 |
| PMMFNet (Wang et al. 2025) and 6 controlled variants, 4 classes | 0.40–0.44; within-variety AUC 0.51–0.53 |
| GMKL (Hassanzadeh et al. 2024), pixel and leaf level, 4 classes | 0.42 / 0.38; within-variety AUC about 0.5 |
| How much spectral variance variety and disease explain | variety 51%, disease within variety ≤ 1% (SNV) |
| Healthy vs red blotch vs leafroll inside each variety | only Chardonnay healthy vs red blotch differs significantly |
| Red blotch in Chardonnay: 5 feature sets × LR / PLS-DA / gradient boosting × 3 fold designs, with a permutation test | nothing beat the mean-spectrum baseline; within-row AUC 0.54–0.64 |
| Variety logistic regression, raw vs SNV (4 training rows) | 0.920 vs 0.939 |
| Disease CNN design: band z-score vs SNV input, brightness jitter, dropout 0.2, 136 px canvas, capacity test, photo and block folds, 4-row baselines | no change to the conclusion; the CNN memorises training leaves but nothing transfers |
| Variety CNN: mean spectrum fed to the last layer, epochs chosen inside each split | fusion missed the improvement rule set in advance; epoch choice was replaced by one validation row for every model |

## Not tested

- PROSPECT-D through the `prosail` package itself (only a line-by-line port was run).
- Things that need new data: more leaves per vine, imaging later in the season, short-wave infrared bands (Gao et al. 2020).
