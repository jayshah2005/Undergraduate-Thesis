# Hyperspectral/spectral detection of grapevine viral diseases and small-dataset plant-disease classification

Method note: ~20 tool calls. Verified from full text: Sawyer 2023 (Frontiers), Nguyen 2021 (Sensors, Specim IQ), AVIRIS-NG preprint (bioRxiv), NASA DEVELOP EMIT report, wheat bioRxiv (Grinberg 2025). Verified from abstract only: ISHS Sawyer, ASEV abstracts (via search snippets, not fetched in full), Bendel 2020 (snippets only). Several fetches failed (403/404/timeouts); items marked UNVERIFIED rest on search-engine summaries and need checking against the paper.

## 1. Studies detecting GRBV and GLRaV-3 with hyperspectral/multispectral data

### Takeaway
Published accuracy is modest and mostly validated with random/stratified splits: leaf-level binary 80-87% (Sawyer 2023), 4-class 62-77%, GRBV the hardest class; field/canopy 69-77% (ASEV abstracts), UAV 81-87% for GRBV (ASEV abstract), airborne GLRaV-3 85-87% (preprint). Satellite multispectral GLRaV-3 mapping is poor on the minority class. No study found uses held-out-vineyard or held-out-year validation for GRBV/GLRaV-3 classification.

### Cited Findings
**Leaf-level, Fresno State / UC Davis group (Sawyer et al. 2023, Front. Plant Sci. 14:1117869)** — [PMC10036814](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- 496 leaf images from 248 plants; 4 leaves collected per vine, 2 randomly selected for imaging. Cabernet franc (1 vineyard) and Cabernet Sauvignon (2 vineyards), CA. Sampled Aug (pre-symptomatic) and Sept (symptomatic) of 2019 and 2022.
- Class counts (entire / pre-symp / symp): non-infected 135/135/135; leafroll 156/50/106; GRBV 108/86/22; co-infected 97/48/49.
- Sensor: Senop HIS, 40 bands, ~5 nm, 510-710 nm, detached leaves in dark cabinet with LED light, Spectralon reference. Ground truth PCR on petioles; GLRaV-1 to -4 pooled as "leafroll" (GLRaV-3 and GRBV predominant). Reddened PCR-negative leaves labelled non-infected.
- Models: RF on mean 40-band spectrum; 3D-CNN on full hypercube (2 conv blocks, Adam lr 0.001, batch 50, 30 epochs, flips/shifts/scale/rotation augmentation).
- Validation: stratified 5-fold CV, shuffled, image-level splits; no vine grouping described, no held-out vineyard.
- Binary (non-infected vs infected) overall accuracy RF/CNN: entire 79.5/80.9; pre-symptomatic 82.8/85.6; symptomatic 82.4/87.0.
- 4-class overall accuracy RF/CNN: entire 62.2/67.0; pre-symptomatic 77.7/76.9; symptomatic 65.7/73.2.
- 4-class per-class accuracy, entire, RF/CNN: non-inf 67/60; leafroll 74/86; GRBV 35/49; co-inf 67/66. Symptomatic GRBV RF 8% vs CNN 57%; symptomatic co-infection RF 22% / CNN 37%, mostly called leafroll (73% RF, 63% CNN).
- Human expert visual assessment of RGB images: 50.2%/50.8% binary and 39.8%/41.9% four-class (chance 25%); authors say comparison not rigorous.
- Pre-symptomatic infected leaves detected from spectra despite no visible symptoms (binary 82.8-85.6%).

**Leaf-level conference version (Sawyer et al., Acta Hortic. 1390)** — [ISHS](https://ishs.org/ishs-article/1390_32/)
- ~440 leaf images, red cultivars, 2 vineyards, 2 dates one month apart, 500-700 nm. RF 68%, CNN 70% (average over healthy/infected classes, likely the multi-class task); 5-fold CV. Distinguishing GLRaV vs GRBV was "relatively difficult". Earlier/smaller version of the 2023 study.

**Canopy/proximal in-field (Fresno State ASEV abstracts; search-engine summary, UNVERIFIED in detail)** — [ASEV field](https://asev.org/abstract/grapevine-virus-detection-using-field-hyperspectral-images-and-machine-learning-models), [ASEV VIS/NIR](https://asev.org/abstract/identifying-red-blotch-and-leafroll-viruses-visnir-hyperspectral-images-acquired-ground)
- Tripod canopy imaging with 230-band camera (510-900 nm), Rutherford CA, 2020-2021, >700 vines PCR-verified; best 73.3% accuracy using 23 bands on all data, 76% using 30 bands on the symptomatic subset; pre-symptomatic stage 68.6% rising to 76.6% (SVM) with symptoms (as reported by search summary of the abstract; I could not open the abstract pages directly, they returned 404 on fetch).

**UAV, GRBV (ASEV abstract, Laroche-Pinel/Brillante group; search summary)** — [ASEV UAV](https://asev.org/abstract/scalable-vine-level-assessment-grapevine-red-blotch-virus-infections-aerial-hyperspectral)
- 3-ha Cabernet franc vineyard, Oct 2021, drone camera with 25 bands, 520-820 nm; 264 randomly selected PCR-tested vines; six ML models. Overall accuracy 87.0% with raw spectra vs 81.4% using PROSPECT-inverted leaf pigments as predictors; highest feature importance for estimated anthocyanin. Validation scheme not stated in the snippet; with 264 vines in one vineyard/one date this is a single-site result. (Page fetch failed with 404; figures from search summary.)

**Airborne imaging spectroscopy, GLRaV-3 (bioRxiv 2022, AVIRIS-NG, Lodi CA)** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- AVIRIS-NG, 425 bands 380-2510 nm, flown 18 Sep 2020; resampled 1/3/5 m. Cabernet Sauvignon subset ~204 ac.
- Pixel counts (1 m): non-infected ~621,000; asymptomatic 2,258; symptomatic 1,027. Scouts found 1,427 (2020) / 2,398 (2021) infected vines; lab confirmation of only 100 (2020) and 10 (2021) cane samples.
- RF with SG smoothing + PCA(10) + SMOTE; 70/30 split with 10-fold CV averaged over 10 models. Best (3 m): non-infected vs asymptomatic 87% accuracy (Kappa 0.73); non-infected vs asymptomatic+symptomatic 85% (Kappa 0.71). No AUC. 3 m not significantly better than 1 or 5 m.
- Asymptomatic labels inferred from 2021 scouting (not molecular), non-infected assumed beyond 5 m buffer; confusion at vineyard edges and from abiotic stress.

**Satellite, GLRaV-3 (NASA DEVELOP, Lodi, 2022-2024)** — [NASA NTRS PDF](https://ntrs.nasa.gov/api/citations/20250011141/downloads/2025Fall_ARC_LodiAg_TechPaper_FD-Final.pdf)
- EMIT (hyperspectral, 60 m pixel) RF: overall accuracy ~90.5%, class-1 (>=1% symptomatic vines) precision 24.7%, recall 74.1%, F1 0.37; 20% held out for validation. Sentinel-2 RF: accuracy ~92.4%, class-1 recall 32.5%, precision 38.6%, F1 0.35. Strong interannual variation in EMIT predictions (authors suspect drought/2022). Training: 1,926 class-0 and 76 class-1 pixels before undersampling. Split is random stratified pixel-level; scouted vines labelled by visual symptom scouting from 2017-2025. Accuracy headline is inflated by class imbalance; F1 is the honest metric.

**Ground-based 400-2500 nm GLRaV-1/-3 (Bendel et al., Remote Sens. 12(10):1693, 2020; from search summary only, UNVERIFIED)** — [MDPI](https://www.mdpi.com/2072-4292/12/10/1693)
- Greenhouse models separated symptomatic/asymptomatic/healthy; 3 years of field tests gave high detection of symptomatic white and red cultivars; asymptomatic field vines less reliable; simulated multispectral sensor from top wavelengths gave comparable accuracy; GLRaV-1 vs -3 symptoms could be differentiated. Exact accuracies not retrieved (page returned 403).

**Cornell UAS dissertation (search summary, UNVERIFIED)** — [Cornell eCommons](https://ecommons.cornell.edu/entities/publication/dabd5a78-b09b-410a-bed5-6f5a5a570ab6)
- PLSR on VIS-NIR UAS imagery detecting GLRaV-3 symptoms in presence of downy mildew: F1 0.69-0.72 for symptomatic vines, best post-veraison; separating mildew vs GLRaV-3 accuracy 0.55-0.82.

### Inferences
- Best-supported numbers for GRBV vs GLRaV discrimination are from one group (Fresno State); results across independent groups are not available.
- Whole-canopy/field studies sit at ~70-77%, below leaf-level binary tasks, consistent with extra noise (illumination, soil, mixed pixels).

### Gaps
- No full text obtained for Fresno State field and UAV papers (only abstracts via search summaries); validation scheme there unknown.
- No GRBV or GLRaV-3 study found using Specim IQ (400-1000 nm, 204 bands); the Specim IQ grapevine studies found are for GVCV (Nguyen 2021) and esca/Armillaria (below).
- No AUC values reported in any retrieved GRBV/GLRaV-3 paper.

## 2. Validation schemes, leakage and inflated results

### Takeaway
Every retrieved GRBV/GLRaV paper uses random or stratified image/pixel-level CV or holdout; none documents grouping by vine, row, vineyard or year. Where an independent hold-out exists (Nguyen 2021), accuracy collapses.

### Cited Findings
- Sawyer 2023: stratified 5-fold, image-level, two leaves per vine, so same-vine leaves can straddle train/test; the paper does not describe vine grouping and does not quantify it (my reading of the Methods as extracted) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- AVIRIS-NG preprint: 70/30 + 10-fold on pixels; multiple 1 m pixels per vine possible; no vineyard grouping described — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- NASA/EMIT: random stratified 70-30 pixel splits plus 20% held-out validation (random) — [NASA](https://ntrs.nasa.gov/api/citations/20250011141/downloads/2025Fall_ARC_LodiAg_TechPaper_FD-Final.pdf)
- Nguyen 2021 (GVCV, Specim IQ, 40 vines; 24 train/16 test images by date): pixel-based RF validation (5-fold CV) 94.7% / 95.3% / 91.6% / 89.7% (Aug 7 / Aug 29 / Sep 19 / Oct 8) but test accuracy 77.75% / 41.89% / 28.71% / 58.80%; combined 85.1% CV vs 73.62% test. 2D/3D-CNN features + RF: CV 71%/75%, test 50% (8/16 images). VI-based SVM CV 82-97% vs test 65.7-67.8%. Test below 50% chance on two dates. — [PMC7866105](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866105)
- General remote-sensing evidence that random CV overestimates: simulated hyperspectral LAI study (Remote Sens. 2018) found CV accuracy overestimated when spatial structure is learned; a drone-CNN tree study found random holdouts overestimated by up to 28% vs spatial blocks (search summaries, not opened; counterpoint: a Wageningen paper argues the bias is not always spatial autocorrelation). — [DOAJ](https://doaj.org/article/75f63f28295c4f9b97cb80003757524d), [Wageningen](https://edepot.wur.nl/569552)

### Inferences
- Nguyen 2021 is the clearest in-domain quantification of CV-vs-holdout inflation (about 10-65 points depending on date). Expect similar optimism in leaf-level GRBV/GLRaV numbers that split by image.
- Leaf/pixel from same vine in train and test is the specific leak to avoid; group by vine at minimum, preferably by vineyard/year.

### Gaps
- No quantified effect of vine-grouped vs image-level splits in GRBV/GLRaV data.

## 3. Small-dataset hyperspectral plant disease: deep vs classical

### Takeaway
Evidence is mixed and thin. Where small hyperspectral sets (tens to hundreds of samples) are used, CNN results are not clearly better than RF/SVM, and pretrained 2D backbones with augmentation work best for modest-size sets. Classical methods remain strong baselines.

### Cited Findings
- Sawyer 2023 (496 images): 3D-CNN beat RF by 1-5 points in binary, by up to 5-7 on 4-class entire/symptomatic, but RF was better on 4-class pre-symptomatic (77.7 vs 76.9); RF used mean spectra only, CNN used the full hypercube, so input differs. Symptomatic GRBV (n=22): CNN 57% vs RF 8%. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- Sawyer ISHS (~440 images): RF 68% vs CNN 70%. — [ISHS](https://ishs.org/ishs-article/1390_32/)
- Nguyen 2021 (40 vines): image-based 2D/3D-CNN features + RF reached 50% on holdout, while pixel-based PCA+RF reached 73.6% on combined holdout; deep image-level models were worse than pixel-level classical ones. — [PMC7866105](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866105)
- Wheat multi-infection (Grinberg, bioRxiv 2025): pretrained EfficientNet-B0 / Inception-V3 with a learned 3-channel input layer (2D or 3D conv), 3-fold CV; best 2D-EfficientNet 81% overall accuracy, 72% on yellow rust+mildew; Septoria and co-infection classes poor due to few samples; no SVM/PLS-DA baseline reported (as seen in extracted text). — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2025.07.16.664902.full.pdf)
- Wheat rust unbalanced data (Cuenca-Romero 2024, Front. Plant Sci. 10.3389/fpls.2024.1392409): SVM and RF beat ANN, especially with SMOTE; yellow rust RF 70% without augmentation, brown rust SVM 63% with SMOTE (search summary, not opened). — [PMC11130404](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11130404/)
- AVIRIS-NG preprint used RF with SMOTE and PCA, no deep model. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- Tomato frost few-shot paper (Agronomy 13(9):2348) compared SVM/PLS-DA, CNN, Prototypical and Relation Networks; results not retrieved (403). — [MDPI](https://www.mdpi.com/2073-4395/13/9/2348)
- Other specim IQ grapevine small studies: esca, 11 Tempranillo vines, 200 pixels, PLS-DA 97% in cross-validation (search summary); Armillaria (Calamita 2021, Teroldego) LDA on VIs/bands 85%/84% (search summary). Pixel-based CV on very few vines. — [Esca (UPNA)](https://academica-e.unavarra.es/entities/publication/395eba21-5b95-48bc-88e9-f771c21362b9), [Calamita 2021 DOAJ](https://doaj.org/article/be68020273bf4f53baeeb2546adf58df)

### Inferences
- At <=500 samples and without vine-grouped validation, CNN-vs-RF differences of 1-5 points are within what split choice could shift; no study ran repeated grouped CV with confidence intervals.
- Pretrained 2D backbones (wheat study) are the only transfer-learning-style evidence retrieved; no hyperspectral grapevine paper tested transfer learning.

### Gaps
- No clean head-to-head (PLS-DA/SVM vs CNN) on a <=2000-sample hyperspectral disease set with grouped validation was retrieved; tomato frost paper would be the most direct but wasn't readable.
- Did not search general reviews of transfer/self-supervised learning for hyperspectral crops.

## 4. Informative wavelengths and indices

### Takeaway
Signal sits in the visible (yellow-orange-red, ~530-700 nm: anthocyanin/chlorophyll) and red-edge onset; in canopy/airborne data NIR/SWIR also contribute. Pre-symptomatic signal differs from symptomatic signal.

### Cited Findings
- Sawyer 2023 RF importance: pre-symptomatic peaks at 586 and 596 nm; symptomatic ~530, 600, 650 and ~700 nm (red-edge onset). Authors link to chlorophyll absorption (~626/642 nm) and anthocyanin reddening, interpretive, no pigment measurements. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- UAV GRBV: estimated leaf anthocyanin (from PROSPECT inversion) had highest feature importance, but raw spectra (87.0%) outperformed PROSPECT outputs (81.4%) (ASEV summary, unverified). — [ASEV](https://asev.org/abstract/scalable-vine-level-assessment-grapevine-red-blotch-virus-infections-aerial-hyperspectral)
- AVIRIS-NG: authors attribute performance to non-visible wavelengths (search summary) but report no specific bands; noisy water-absorption regions (1310-1470, 1750-2000, 2400-2600 nm) excluded. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- GVCV on Specim IQ (Nguyen 2021): 900-940 nm NIR significant at 30 days after start; 449-461 nm blue at 90 days; useful indices NPQI, FRI1, PSRI, AntGitelson early, WSCT and WI (water indices) later; index stability affected by viewing angle, time of day, sunlight. — [PMC7866105](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866105)
- NASA/EMIT work used PCA features (10 components) for EMIT and indices incl. PSRI/EVI for Sentinel-2. — [NASA](https://ntrs.nasa.gov/api/citations/20250011141/downloads/2025Fall_ARC_LodiAg_TechPaper_FD-Final.pdf)

### Inferences
- A sensor limited to 400-1000 nm (Specim IQ) covers the visible/red-edge/NIR bands that carried signal in Sawyer and UAV GRBV work, but not SWIR water bands used in AVIRIS-NG/Bendel.

### Gaps
- No Bendel (2020) selected-wavelength list retrieved; no consensus GRBV-specific vs GLRaV-3-specific band set.

## 5. Confounding: variety, cultivar, symptomatic vs asymptomatic

### Takeaway
Cultivar is mostly uncontrolled or acknowledged but not tested; symptom stage strongly changes both accuracy and which classes are separable; abiotic stress and mixed pixels confound canopy-level results.

### Cited Findings
- Sawyer 2023 pooled Cabernet franc and Cabernet Sauvignon across 3 vineyards; authors note cultivar could act as an unintended cue; sampling dates differ in class composition (pre-symptomatic and symptomatic subsets have different class sizes), confounding stage comparisons; symptomatic GRBV only 22 images. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/)
- Pre-symptomatic vs symptomatic: Sawyer binary accuracy 82.8-85.6 (pre) vs 82.4-87.0 (symp); 4-class RF better pre-symptomatic (77.7) than symptomatic (65.7). Field canopy: 68.6% pre-symptomatic vs 76.6% symptomatic (ASEV summary). AVIRIS-NG: most confusion between symptomatic and asymptomatic. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036814/), [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- AVIRIS-NG: only Cabernet Sauvignon analysed so variety effect not tested; abiotic stress and vineyard-edge effects drive false positives; asymptomatic labels not molecularly confirmed. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.04.510827v1.full)
- Bendel 2020: red vs white cultivars treated separately in field tests; asymptomatic field detection less reliable (search summary). — [MDPI](https://www.mdpi.com/2072-4292/12/10/1693)
- Cornell: GLRaV-3 detection complicated by downy mildew; separation accuracy 0.55-0.82 (search summary). — [Cornell](https://ecommons.cornell.edu/entities/publication/dabd5a78-b09b-410a-bed5-6f5a5a570ab6)
- NASA: interannual variation (drought 2022) shifts EMIT predictions. — [NASA](https://ntrs.nasa.gov/api/citations/20250011141/downloads/2025Fall_ARC_LodiAg_TechPaper_FD-Final.pdf)
- Nguyen 2021: single cultivar (Chardonel), one season; viewing angle/time of day/sunlight confound indices. — [PMC7866105](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866105)

### Inferences
- Cultivar-by-vineyard-by-class confounding is likely in leaf-level datasets (a vineyard with mostly infected vines would let the model learn the site); only leave-vineyard-out validation would reveal it.

### Gaps
- No study retrieved that explicitly tests cultivar as a confounder (e.g., leave-cultivar-out) for GRBV/GLRaV-3.
- No data on within-vine leaf-position or leaf-age effects.
