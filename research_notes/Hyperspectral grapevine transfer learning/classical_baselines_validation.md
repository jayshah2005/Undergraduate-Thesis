# Classical chemometric/ML baselines vs deep learning for small hyperspectral leaf datasets, and rigorous validation

Method note: ~14 tool calls (web search plus page fetches). Where a number comes from a fetched full text it is marked as such; numbers from search snippets only are flagged "(snippet only)". No source found directly answers several questions (see Gaps). Several sources are 2026 preprints/papers; one arXiv ID (2610.08770) was returned by the fetch tool and not independently confirmed.

## 1. Head-to-head: PLS-DA/SVM/GBM vs CNN/transformer on small hyperspectral plant datasets

### Takeaway
On small leaf-level disease datasets, tuned classical models (SVM, XGBoost, ANN on selected bands) land within ~1-2 points of a transformer, and PLS-DA and RF are often noticeably worse. Few papers do a fair matched-validation comparison, so "deep learning wins" claims are weakly supported. Use tuned SVM (RBF), logistic regression, and gradient boosting on mean/pixel spectra as the bar. PLS-DA is a weaker, though traditional, baseline.

### Cited Findings
- Tobacco leaf disease (95 leaves: brown spot 35, wildfire 33, PVY 14, TMV 8, healthy 5; 5,000 pixel spectra/class sampled after a leaf-level split). Test accuracy with SG preprocessing, full 462 bands: ANN 98.93%, SVM 97.90%, XGBoost 97.50%, PLS-DA 88.47%, RF 82.67%; Transformer 99.35% (about 4.84M parameters). With 20 SPA bands: ANN 98.88%, XGBoost 97.01%, SVM 96.23%, PLS-DA 87.09%, RF 80.44%. The authors note small leaf counts for TMV, PVY and healthy as a limitation. — [Frontiers in Plant Science 2026, PMC12816246](https://pmc.ncbi.nlm.nih.gov/articles/PMC12816246/)
- Grapevine virus (red blotch/leafroll), ~500 leaves from 250 vines, 510-710 nm: 3D CNN 87% vs RF 82.8% (binary); multiclass 77.7% (CNN) vs 76.9% (RF). The snippet did not show whether CV was split by leaf or vine. — [Frontiers in Plant Science 2023](https://www.frontiersin.org/articles/10.3389/fpls.2023.1117869/full) (snippet only)
- Same group's field-canopy work compared SVM, PLS-DA, RF; best overall accuracy 73.3% with 23 bands of 8 nm. — [ISHS](https://ishs.org/ishs-article/1395_27/) (snippet only)
- Wheat co-infection (1,447 images, 6 classes): best was 2D EfficientNet-B0 at 0.806 OA (3D EfficientNet 0.768, 2D Inception-V3 0.762, 3D Inception-V3 0.697). It reported NO classical baseline and only cites external work saying CNNs beat RF/SVM. So it provides no evidence of CNN > classical. — [Wheat multiple infections, PMC12595906](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595906/)
- Wheat/barley stem rust: linear SVM, logistic regression (L1/L2) and LightGBM on first-order-derivative features gave best F1 of 0.962 (wheat) and 0.94 (barley); channel-wise standardization gave the largest F1 gain; wheat->barley infected recall 0.986, barley->wheat infected recall 0.981 (control recall 0.941 / 0.892). Text has internal inconsistencies (F1 0.962 vs 0.94 test). — [Cross-crop stem rust, PMC12608451](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608451/)
- Banana black sigatoka: SVM, MLP, NPLS-DA, PLS-PLR all detected disease with high accuracy (no CNN). — [Plants 2022](https://doaj.org/article/91b25a992c434f2eb03ec2999299428d) (snippet only)
- Strawberry leaf: SVM on full spectra best among SAM/SVM/KNN/LDA, 94%. — [E3S Web Conf. 2023](https://www.e3s-conferences.org/10.1051/e3sconf/202339003018) (snippet only)
- Non-disease spectroscopy evidence for small data: in a Raman concentration benchmark PLS was best on 40/48 datasets, but a simple 1D CNN beat it on higher-noise datasets with 10,000 points. — [arXiv 2306.16621](https://arxiv.org/pdf/2306.16621) (snippet only)
- Mishra and Passos: DL has been shown to outperform PLS for single fruit-trait NIR prediction, but with augmentation because DL needs larger datasets; PLS is sometimes sufficient. — [Perspectives on DL for NIR](https://research.wur.nl/en/publications/perspectives-on-deep-learning-for-near-infrared-spectral-data-mod/) (snippet only; regression, not classification)
- Lettuce trait regression: 2D CNN / FC nets had higher Rp2 than five multivariate methods (not framed around small n; regression). — [Frontiers Plant Sci 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9279906/) (snippet only)

### Inferences
- In the one fully-read paper with all baselines on the same split, ordering is ANN ~ transformer > SVM ~ XGBoost >> PLS-DA > RF. So a reasonable bar for ~1000 leaves: RBF-SVM, L2 logistic regression, and XGBoost/LightGBM/ANN on SG/SNV-processed mean or pixel spectra. PLS-DA and RF are cheap sanity checks, not the bar. (The tobacco n=95 leaves is far smaller than 1000; absolute numbers will not transfer.)
- Differences of 1-2 points among top models on a ~95-leaf test set are within sampling noise; report CIs over leaf-grouped repeats.
- Deep gains are most plausible where spatial/texture lesion information matters (image-level inputs), not on mean spectra.

### Gaps
- No fair, matched-validation head-to-head (PLS-DA/SVM/GBM vs CNN/transformer, grouped CV) on a ~1000-leaf plant dataset found. Most CNN-vs-RF comparisons (grapevine) don't state whether splits were leaf/vine-level.
- Mishra et al. 2022 TrAC review numbers not retrieved.

## 2. Pixel-level training, leaf-level aggregation, MIL: evidence and pitfalls

### Takeaway
Pixels from the same leaf are pseudo-replicates; the leaf (or plant) must be the split and inference unit, with pixel predictions aggregated (mean probability/vote) per leaf. Evidence on MIL for plant hyperspectral work is thin.

### Cited Findings
- Tobacco study split at leaf level before pixel sampling, so all pixels of a leaf stay in one subset; then sampled 5,000 pixels per class. — [PMC12816246](https://pmc.ncbi.nlm.nih.gov/articles/PMC12816246/). Caveat: Section 2.7 describes 5-fold stratified CV without explicitly saying it is leaf-grouped (per fetch summary).
- Flavescence doree on Pinot Noir: per-pixel vegetation indices aggregated per leaf, 92% accuracy. — [Univ. Brescia record](https://iris.unibs.it/handle/11379/629305) (snippet only)
- A MIL study of leaf wilt in soybean (Banerjee et al., Agriculture 2025) exists, but only the author list was seen; method and results unverified. — [IDEAS](https://ideas.repec.org/a/gam/jagris/v15y2025i6p614-d1611767.html)
- Pixel/patch leakage in HSI (Pavia University, six models incl. SVM, MLP, 2D/3D-CNN, ViT, MorpMamba): 3D-CNN OA 96.17% under random sampling vs 55.20% under non-random spatial sampling; ViT and 2D-CNN lose 40.71 and 38.81 points; overlap of test patches with training grows from 23.28% (5x5) to 77.02% (19x19). — [Alkhatib, arXiv 2610.08770 (2026 preprint, not peer reviewed)](https://deeplearn.org/arxiv/841523/data-leakage-in-patch-based-hyperspectral-image-classification:-quantifying-the-impact-of-spatial-overlap)
- Pixel-wise random sampling is the most common partition and yields substantial train/test overlap; regions with larger overlap show higher accuracy. — [SaaFormer, arXiv 2306.16759](https://arxiv.org/abs/2306.16759v1)
- Disjoint-sampling proposal for HSI benchmarks (Indian Pines, Pavia U, Salinas, Houston, Botswana); abstract-level claim that overlapping splits inflate metrics, but no disjoint-vs-random numbers retrievable. — [arXiv 2404.14944](https://huggingface.co/papers/2404.14944.md)
- Raman analogue: split at highest hierarchical level (biological replicate) and repeat dimension reduction inside each CV loop. — [Guo et al. 2017, Anal. Methods](https://pubs.rsc.org/en/content/articlepdf/2017/ay/c7ay01363a) (snippet only)

### Inferences
- Remote-sensing leakage (pixels in one scene) is the same mechanism as pixels in one leaf, only more extreme: neighbouring pixels within a leaf are near-duplicates. Random pixel splits on a leaf dataset will likely inflate accuracy heavily (magnitude on leaf data not measured here).
- Pixel samples should be class-balanced per leaf (cap pixels per leaf) so large leaves do not dominate loss and pixel-count weighting does not masquerade as leaf-level performance.
- Fit scalers, SPA/CARS band selection, PCA inside training folds only (supported by Guo et al.; tobacco study fit z-score on train only).

### Gaps
- No quantified pixel-split vs leaf-split inflation on a plant-leaf dataset found.
- No verified MIL-vs-mean-aggregation comparison for leaf hyperspectral data.

## 3. Preprocessing that matters

### Takeaway
No consistent winner; SG smoothing/first derivative and channel-wise standardization are the safest defaults, and raw spectra are sometimes just as good. Treat preprocessing as a hyperparameter selected inside grouped CV.

### Cited Findings
- Stem rust: first-order derivatives with linear SVM were best (F1 0.962); channel-wise standardization gave the largest F1 gain; SG smoothing and SNV on averaged curves were also used. — [PMC12608451](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608451/)
- Tobacco: compared SG, SNV, MMS, SS; SG chosen as primary; band selection SPA (20 bands) retained 98.88% for ANN but hurt SVM (97.90 -> 96.23), XGBoost modestly, PLS-DA (88.47 -> 87.09), and CARS reduced SVM to 91.79%. — [PMC12816246](https://pmc.ncbi.nlm.nih.gov/articles/PMC12816246/)
- PVY virus leaves: SG + SVM best, 98.1% average precision on severity levels; vs MSC and SNV. — [Frontiers Plant Sci 2023](https://doaj.org/article/dfa0e04efacd4c0487fe7a7b2d4a4d94) (snippet only)
- Solanaceae leaves: SG smoothing had better classification accuracy than 1st/2nd derivatives in one study; a 2025 study had best SVM (84.7%) after first derivative. Tea leaves: SVM on raw spectra was most robust. Conflicting. — [Spectrosc. Spectral Anal. 2024](https://m.researching.cn/articles/OJ5ccc40ec4dbf3ead); [2025](https://www.researching.cn/articles/OJ29c6e4cdd8b4cb11); [Frontiers 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.940327/pdf) (snippets only)
- Oilseed rape: direct standardization reduced spectral differences between sample sets (calibration transfer). — [PMC6021932](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6021932/) (snippet only)
- Stem-rust pipeline background segmentation via k-means (k=2) on pixels before analysis. — [PMC12608451](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608451/)

### Inferences
- Candidate grid: {white/dark reference reflectance calibration (mandatory), SG(w, order 2) smoothing, SG 1st derivative, SNV, per-band standardization fit on train} x {all bands, SPA/CARS}. Band selection helps ANN/XGBoost less-damagingly than SVM/PLS-DA in the tobacco data, so evaluate per model.
- Per-session/day reference calibration and batch effects (direct standardization) matter if leaves were imaged on different days; this ties to label confounding (Section 4).

### Gaps
- No controlled SNV vs SG-derivative comparison with illumination variation on leaves found; no source on illumination-geometry normalization specific to leaf imaging.

## 4. Validation pitfalls: grouped CV, leave-variety-out, confounding, imbalance, measured inflation

### Takeaway
Group at the highest independent unit (plant/vine, then variety/session) for any reported number. The only measured random-vs-spatial gap I found is from remote-sensing HSI (up to ~41 points); for leaf/plant data the size is unmeasured in my sources but the mechanism is identical.

### Cited Findings
- Measured inflation (Pavia University): 96.17% -> 55.20% for 3D-CNN, 40.71 and 38.81 point losses for ViT/2D-CNN. — [arXiv 2610.08770](https://deeplearn.org/arxiv/841523/data-leakage-in-patch-based-hyperspectral-image-classification:-quantifying-the-impact-of-spatial-overlap)
- Controlled-random sampling reduces but does not remove overlap (boundary pixels); random splits give inconsistent results across draws. — [arXiv 2302.05297](https://arxiv.org/pdf/2302.05297) and [2404.14944](https://huggingface.co/papers/2404.14944.md) (snippets)
- Spectroscopy replicates (same aliquot/spot) appearing in train and test make random k-fold optimistic; CV can underestimate error, permutation testing recommended; use repeated grouped 5-fold with the lab-sample combination as unit. — [Basque thesis 2025](https://www.nanogune.eu/eu/argitalpenak/phd-theses/224); [RSC Analyst-style paper](https://pubs.rsc.org/en/content/getauthorversionpdf/C4AN01783H); [arXiv 2609.17963](https://arxiv.org/pdf/2609.17963) (snippets only)
- Cross-crop test as a stricter validation: wheat->barley and barley->wheat infected recall 0.986 / 0.981 but control recall 0.941 / 0.892, i.e. false positives rise on the unseen domain. — [PMC12608451](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608451/)
- Cross-variety/genotype generalization is studied (rice, four varieties with transfer learning, 2021; N. benthamiana, four genotypes, 3D-CNN, Plant Methods 2025) but I did not retrieve numbers. — [Frontiers 2021](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.693521/epub); [Plant Methods 2025](https://biomedcentral-plantmethods.publicaciones.saludcastillayleon.es/articles/10.1186/s13007-025-01337-0)
- Wheat co-infection used plain 3-fold CV with no stated grouping and class counts of 36 (Septoria) and 39 (YR+Septoria) vs ~360 for others. — [PMC12595906](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595906/)
- Tobacco: test class counts tiny (TMV 8 leaves, healthy 5), acknowledged as limitation. — [PMC12816246](https://pmc.ncbi.nlm.nih.gov/articles/PMC12816246/)

### Inferences
- Recommended protocol: nested, repeated, grouped CV (group = plant/vine; stratify by label; additionally leave-one-variety-out and leave-one-session/date-out as a stress test); report macro-F1, per-class recall, balanced accuracy, and leaf-level (not pixel-level) metrics with bootstrap CIs over groups; permutation test; compare against a majority-class and a "variety-only"/"session-only" predictor to detect confounding.
- If disease label correlates with variety, a classifier can learn variety. Leave-variety-out accuracy minus random-split accuracy is the diagnostic. Not directly measured in found sources.
- Class imbalance: use class weights and macro metrics; avoid pixel-count-based balancing across leaves.

### Gaps
- No source quantifying random vs grouped-by-plant inflation on leaf hyperspectral data.
- No retrieved numbers for leave-one-variety-out drops.
- Variety-confounding is argued from first principles here, not from a cited study.

## 5. Multi-label handling (two diseases per leaf)

### Takeaway
Little direct evidence. The one co-infection hyperspectral paper found treated it as multi-class with explicit combination classes, which is data-hungry for rare combinations.

### Cited Findings
- Wheat: six mutually exclusive classes including YR+mildew (376 images) and YR+Septoria (39); no Septoria+mildew class (not producible); best 2D EfficientNet-B0 OA 0.806, 72% on YR+mildew (snippet). Explicitly multi-class, chosen to capture pathogen interactions. — [PMC12595906](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595906/); [Cambridge record](https://www.repository.cam.ac.uk/handle/1810/392101)
- Remote-sensing multi-label HSI: ML-KNN and binary relevance (BR) performed poorly relative to a proposed method (not plant disease). — [cjig.cn](https://www.cjig.cn/zh/article/doi/10.11834/jig.190285/) (snippet only)

### Inferences
- Standard options (from general ML knowledge, not from a retrieved source): binary relevance (one sigmoid/logistic per disease), label powerset (as in the wheat paper), classifier chains. With ~1000 leaves and few double-infected leaves, binary relevance with per-label class weights, evaluated with per-label F1/AUC plus subset accuracy on co-infected leaves, is the natural baseline; stratify grouped folds on label combinations (iterative stratification). Note that combined-infection spectra may not be additive, which limits BR.

### Gaps
- No plant hyperspectral study comparing BR vs label powerset vs chains found; no best-practice source retrieved. Iterative-stratification claim is unsourced.
