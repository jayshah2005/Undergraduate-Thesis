# Archived: transfer-learning study (47 runs + 16 stress-test runs, 2026-10-18)

The first round of approaches on the 998 leaves (labels: red blotch, leafroll 3). Archived on 2026-10-19 to keep
`claude runs/experiments/` clean for paper implementations. None of the approaches did well.

## Setup shared by every run

- Unit = one leaf (998 leaves, 4 varieties). Every model outputs two probabilities per leaf: P(red blotch) and
  P(leafroll 3), trained as two separate yes/no questions.
- 5 folds: fold k = every leaf in row R(k+1) of all four varieties, so whole rows are held out (neighbouring vines
  look alike). Every leaf is predicted by the model that never saw its row. The leave-one-variety-out runs are the
  only exception (4 folds = the 4 varieties).
- Leafroll is neither trained nor scored on Cab Franc (1 positive leaf out of 250).
- Pixels never cross folds. Scalers and per-band standardisation are fitted on training leaves only.
- No hyperparameter tuning: fixed defaults, fixed number of epochs, no early stopping, one random seed per neural
  network.
- Class imbalance: `class_weight="balanced"` in scikit-learn models; in neural networks the positive class is
  up-weighted by #negatives / #positives per label (`pos_weight`).
- Neural networks share one training loop: AdamW (weight decay 0.01), learning rate decayed to ~0 along a cosine,
  gradients clipped at norm 5, binary cross-entropy per label.

**Scores in the tables.** *Within-variety ROC AUC* compares a leaf only with leaves of the same variety and fold, so
0.5 = chance and recognising the variety does not help; each value carries about ±0.05 (95% bootstrap).
*Balanced accuracy* is at probability 0.5 and pooled over all leaves, so it is inflated by the variety: the variety
prior alone reaches 0.652 for red blotch.

## Overview: best variant of each approach

| Approach | Runs | Within-variety ROC AUC<br>any disease / red blotch / leafroll | Balanced accuracy<br>red blotch / leafroll |
|---|---|---|---|
| 01 Variety prior: each leaf gets its variety's disease rate (no model) | 1 | 0.500 / 0.500 / 0.500 | 0.652 / 0.491 |
| 02 One summary spectrum per leaf (mean raw / SNV / 1st derivative, or mean+std+p10+p90) × logistic regression, RBF-SVM, PLS-DA, gradient boosting; one model for all varieties | 16 | 0.558 / 0.580 / 0.529 | 0.695 / 0.656 |
| 02 Same 16 combinations, a separate model inside each variety | 16 | 0.581 / 0.589 / 0.527 | 0.658 / 0.604 |
| 02 Same 16 combinations, trained on 3 varieties and tested on the 4th | 16 | 0.553 / 0.552 / 0.507 | 0.537 / 0.471 |
| 03 Every pixel a sample (204 bands, leaf's label), logistic regression or gradient boosting, leaf = mean of its pixels | 2 | 0.542 / 0.584 / 0.485 | 0.694 / 0.649 |
| 04 Same pixel set-up with an MLP or a 1-D CNN along the wavelength axis | 2 | 0.538 / 0.582 / 0.503 | 0.695 / 0.671 |
| 05 Leaf = bag of 256 pixels; attention weights per pixel and per disease (attention MIL) | 1 | 0.527 / 0.570 / 0.482 | 0.692 / 0.661 |
| 06 Whole-leaf image, 204→3 band adaptor, ResNet-18 from ImageNet or from scratch, plus a fixed pseudo-RGB control | 3 | 0.574 / 0.560 / 0.496 | 0.677 / 0.584 |
| 07 Whole-leaf image into DOFA (hyperspectral, all bands + wavelengths) or DINOv2 (RGB + adaptor), frozen or last 2 blocks tuned | 4 | 0.587 / 0.565 / 0.544 | 0.681 / 0.662 |
| 08 Masked-band autoencoder pretraining on unlabelled pixels, then the pixel classifier of 04; vs same net from scratch | 2 | 0.557 / 0.576 / 0.498 | 0.707 / 0.671 |

Each number is the best over that approach's variants, picked per column, so it is optimistic. Every variant is
listed below.

## What was tried, approach by approach

### 01 Variety prior (`01_baseline_variety_prior.py`)

- No model and no spectra. For each test leaf, the prediction is the fraction of diseased leaves of its variety in the
  training folds (e.g. every Cab Franc leaf gets P(red blotch) ≈ 0.96), separately for red blotch and leafroll.
- Purpose: the bar any model has to beat. It scores exactly 0.5 within variety by construction, and shows how much a
  pooled score can come from the variety alone (pooled AUC 0.745 red blotch, 0.622 leafroll).

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 01_variety_prior | 0.500 / 0.500 / 0.500 | 0.652 / 0.491 |

### 02 Classical models on one summary spectrum per leaf (`02_classical_leaf_level.py`)

Each leaf (thousands of pixels) is reduced to one row of numbers, which goes through a `StandardScaler` (fitted on
training leaves) and a classifier. A separate model is trained for red blotch and for leafroll. 4 feature sets × 4
models = 16 combinations, each run three ways (48 runs).

Feature sets:
- `mean_raw`: mean reflectance of each band over all leaf pixels (204 values).
- `mean_snv`: the mean spectrum after Standard Normal Variate: each spectrum centred and divided by its own standard
  deviation, which removes overall brightness differences (204 values).
- `mean_deriv`: Savitzky–Golay first derivative of the mean spectrum, window 11 bands (~33 nm), 2nd-order polynomial;
  removes baseline offsets and emphasises curve shape such as the red edge (204 values).
- `stats_raw`: mean, standard deviation, 10th and 90th percentile of each band over the leaf's pixels (816 values).

Models:
- `logreg`: logistic regression, L2 penalty, C = 0.1 (fairly strong regularisation), balanced class weights.
- `svm`: support vector machine, RBF kernel, C = 1, gamma = "scale", balanced class weights; the score is the sigmoid
  of the distance to the decision boundary.
- `plsda`: PLS-DA: partial least squares regression on the 0/1 label with 8 components, output clipped to [0, 1].
- `hgb`: histogram gradient boosting, 200 trees, learning rate 0.05, maximum depth 3, balanced class weights.

Three ways of running the 16 combinations:
- **Pooled:** one model for all four varieties, on the standard 5 row folds.
- **Per variety:** a separate model trained inside each variety on that variety's training leaves. When a variety
  had fewer than 10 training leaves of either class, it got its training disease rate instead; this happened for
  Cab Franc red blotch (9 healthy leaves in total).
- **Leave-one-variety-out:** one pooled model trained on three varieties and tested on the fourth (4 folds), to test
  whether anything carries over to a variety the model has never seen.

Within-variety ROC AUC (any disease / red blotch / leafroll):

| features | model | pooled | per variety | leave-one-variety-out |
|---|---|---|---|---|
| mean_raw | logreg | 0.535 / 0.533 / 0.486 | 0.545 / **0.589** / 0.515 | 0.541 / 0.519 / 0.470 |
| mean_raw | svm | 0.541 / 0.568 / 0.501 | 0.541 / 0.570 / 0.497 | 0.552 / 0.529 / 0.490 |
| mean_raw | plsda | 0.551 / 0.541 / 0.519 | 0.537 / 0.549 / 0.527 | 0.501 / 0.471 / 0.463 |
| mean_raw | hgb | 0.483 / 0.520 / 0.472 | 0.475 / 0.495 / 0.498 | 0.502 / 0.495 / 0.455 |
| mean_snv | logreg | 0.526 / 0.541 / 0.498 | 0.570 / 0.564 / 0.510 | 0.496 / 0.479 / 0.462 |
| mean_snv | svm | 0.530 / 0.549 / 0.454 | 0.558 / 0.556 / 0.491 | 0.479 / 0.466 / 0.462 |
| mean_snv | plsda | 0.533 / 0.530 / 0.529 | **0.581** / 0.546 / 0.509 | 0.499 / 0.466 / 0.481 |
| mean_snv | hgb | 0.517 / 0.514 / 0.491 | 0.531 / 0.528 / 0.490 | 0.497 / 0.502 / 0.488 |
| mean_deriv | logreg | 0.524 / 0.552 / 0.472 | 0.555 / 0.570 / 0.472 | 0.474 / 0.472 / 0.478 |
| mean_deriv | svm | 0.549 / 0.580 / 0.465 | 0.546 / 0.558 / 0.467 | 0.494 / 0.477 / 0.507 |
| mean_deriv | plsda | 0.538 / 0.545 / 0.486 | 0.561 / 0.566 / 0.469 | 0.482 / 0.490 / 0.487 |
| mean_deriv | hgb | 0.509 / 0.545 / 0.519 | 0.525 / 0.557 / 0.495 | 0.487 / 0.472 / 0.478 |
| stats_raw | logreg | 0.512 / 0.554 / 0.439 | 0.533 / 0.545 / 0.496 | 0.541 / 0.546 / 0.417 |
| stats_raw | svm | **0.558** / 0.569 / 0.470 | 0.537 / 0.540 / 0.501 | 0.553 / 0.552 / 0.451 |
| stats_raw | plsda | 0.524 / 0.551 / 0.432 | 0.526 / 0.523 / 0.496 | 0.522 / 0.546 / 0.454 |
| stats_raw | hgb | 0.534 / 0.542 / 0.489 | 0.539 / 0.527 / 0.521 | 0.484 / 0.517 / 0.413 |

Balanced accuracy (red blotch / leafroll):

| features | model | pooled | per variety | leave-one-variety-out |
|---|---|---|---|---|
| mean_raw | logreg | 0.688 / 0.640 | 0.647 / 0.514 | 0.486 / 0.407 |
| mean_raw | svm | 0.680 / 0.650 | 0.653 / 0.480 | 0.446 / 0.460 |
| mean_raw | plsda | 0.626 / 0.521 | 0.656 / 0.604 | 0.484 / 0.406 |
| mean_raw | hgb | 0.657 / 0.567 | 0.602 / 0.559 | 0.442 / 0.393 |
| mean_snv | logreg | 0.682 / 0.621 | 0.635 / 0.545 | 0.471 / 0.428 |
| mean_snv | svm | 0.678 / 0.629 | 0.650 / 0.514 | 0.493 / 0.376 |
| mean_snv | plsda | 0.654 / 0.540 | 0.648 / 0.595 | 0.484 / 0.471 |
| mean_snv | hgb | 0.638 / 0.578 | 0.630 / 0.555 | 0.487 / 0.409 |
| mean_deriv | logreg | 0.674 / 0.595 | 0.637 / 0.508 | 0.503 / 0.460 |
| mean_deriv | svm | 0.685 / 0.654 | 0.654 / 0.521 | 0.495 / 0.359 |
| mean_deriv | plsda | 0.665 / 0.538 | 0.658 / 0.558 | 0.477 / 0.415 |
| mean_deriv | hgb | 0.664 / 0.593 | 0.650 / 0.566 | 0.458 / 0.433 |
| stats_raw | logreg | 0.669 / 0.625 | 0.627 / 0.523 | 0.537 / 0.368 |
| stats_raw | svm | 0.695 / 0.656 | 0.622 / 0.552 | 0.485 / 0.386 |
| stats_raw | plsda | 0.634 / 0.526 | 0.638 / 0.550 | 0.519 / 0.389 |
| stats_raw | hgb | 0.681 / 0.595 | 0.639 / 0.577 | 0.476 / 0.383 |

How it did:
- Pooled models were slightly above chance within variety (best 0.558). Boosting on the plain mean spectrum was
  below chance (0.483, and 0.475 per variety).
- Per-variety models were the best classical option (best 0.581), but not significantly better than pooled.
- Leave-one-variety-out collapsed: pooled AUC fell to 0.38–0.52 for red blotch and 0.29–0.45 for leafroll (below
  chance), because the spectral signal points in opposite directions in Baco and the other varieties.
- No preprocessing (raw, SNV, derivative, extra statistics) won consistently.

### 03 Classical models on individual pixels (`03_pixel_level_classical.py`)

- Every pixel is one training sample with its 204 reflectance values; its label is the label of the leaf it came from.
- 300 random pixels from each training leaf (~240,000 pixels per fold), standardised per band with statistics from
  those training pixels.
- A separate model for red blotch and for leafroll.
- Test leaf = average of the predicted probability over **every** pixel of the leaf.
- `03_pixel_logreg`: logistic regression, C = 0.01, balanced class weights.
- `03_pixel_hgb`: histogram gradient boosting, 150 trees, learning rate 0.1, maximum depth 4, balanced class weights.

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 03_pixel_logreg | 0.524 / 0.529 / 0.485 | 0.687 / 0.640 |
| 03_pixel_hgb | 0.542 / 0.584 / 0.485 | 0.694 / 0.649 |

How it did: no better than the leaf-mean models of 02. Using millions of pixels did not add information, because
the pixels of one leaf are strongly correlated and a diseased leaf contains many healthy-looking pixels.

### 04 Neural networks on individual pixels (`04_pixel_mlp_cnn1d.py`)

- Same pixel set-up as 03: input = one pixel's 204 bands, standardised per band (training-leaf statistics), label =
  the leaf's two labels (one network, two outputs).
- Every epoch draws 100 fresh random pixels from each training leaf.
- Augmentation: brightness ×0.9–1.1, then Gaussian noise (σ = 0.05 in standardised units).
- 15 epochs, batch 512, learning rate 1e-3.
- Test leaf = average probability over every pixel of the leaf.
- `04_pixel_mlp`: fully connected 204 → 256 → 64 → 2, each hidden layer with BatchNorm, ReLU and dropout 0.3.
- `04_pixel_cnn1d`: 1-D CNN sliding along the wavelength axis:
  - three conv layers with 16, 32 and 64 filters (kernel 7, 7, 5), each followed by BatchNorm, ReLU and max-pool ÷2, giving 204 → 25 positions;
  - then flatten (no global pooling, so the wavelength position is kept) → dropout 0.3 → 64 → 2.

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 04_pixel_mlp | 0.522 / 0.564 / 0.482 | 0.695 / 0.671 |
| 04_pixel_cnn1d | 0.538 / 0.582 / 0.503 | 0.691 / 0.668 |

How it did: no better than 02/03. The networks barely fitted even the training data: the final training loss was
0.57–0.60, against 0.64 after the first epoch.

### 05 Attention multiple-instance learning (`05_attention_mil.py`)

- A leaf is a "bag" of 256 random pixels, with a fresh sample every time; each training leaf gives 2 bags per epoch.
- Pixel encoder, shared by all pixels: 204 → 128 → 64 (ReLU, dropout 0.3).
- Attention network: 64 → 32 (tanh) → 2. It gives every pixel one weight per disease, softmax over the 256 pixels,
  so the model can focus on diseased-looking pixels and down-weight healthy-looking ones in a sick leaf.
- The attention-weighted mean of the pixel vectors goes to one linear classifier per disease.
- Same augmentation as 04; 40 epochs, batch 32 bags, learning rate 1e-3.
- Test leaf = average of the predictions for 8 random bags.

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 05_attention_mil | 0.527 / 0.570 / 0.482 | 0.692 / 0.661 |

How it did: no better than plain averaging of pixel predictions (vs `04_pixel_mlp`: +0.005, 95% CI −0.042 to +0.058).
The model found no clearly diseased pixels to attend to.

### 06 ResNet-18 on whole-leaf images, ImageNet-pretrained vs from scratch (`06_resnet_transfer.py`)

- **Input:** each leaf as a 204-band 128 × 128 image.
  - Leaves larger than 128 px were shrunk with mask-aware resizing, so the background does not bleed into the edges; smaller leaves were kept at real size, centred on black.
  - Per-band standardisation from training pixels; the background stays 0.
- **Band adaptor:** a 1×1 convolution, 204 → 3 channels, in front of the network.
  - It is initialised to pick the bands nearest 640, 550 and 460 nm, i.e. a pseudo-RGB picture.
  - It gets a 10× higher learning rate than the backbone.
- **Network:** ResNet-18 with its 1000-class output replaced by 2 outputs.
- **Training:**
  - 25 epochs, batch 32, backbone learning rate 3e-4 (adaptor 3e-3), bf16 mixed precision.
  - Augmentation: random horizontal and vertical flips, random 90° rotations, brightness ×0.9–1.1.
- **Test:** average over 4 views (original, horizontal flip, vertical flip, 180° rotation).
- **The three runs:**
  - `06_resnet18_pretrained`: ImageNet weights (IMAGENET1K_V1), learned adaptor.
  - `06_resnet18_scratch`: the identical network from random weights, so the difference from `pretrained` is pretraining alone.
  - `06_resnet18_pretrained_fixedrgb`: ImageNet weights with the adaptor frozen at the pseudo-RGB bands. This control tests whether learning the band mix helps.

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 06_resnet18_pretrained | 0.574 / 0.535 / 0.496 | 0.655 / 0.584 |
| 06_resnet18_scratch | 0.553 / 0.543 / 0.475 | 0.660 / 0.546 |
| 06_resnet18_pretrained_fixedrgb | 0.561 / 0.560 / 0.489 | 0.677 / 0.569 |

How it did:
- **Pretraining:** pretrained beat scratch on pooled AUC for any disease (+0.047, 95% CI +0.008 to +0.080), but not within variety (+0.021, CI −0.027 to +0.064). The gain came from recognising the variety.
- **Fit vs generalisation:** the pretrained network fitted the training leaves much better (training loss 0.20 vs 0.375, fold 0), but this did not carry over to new leaves.
- **Band adaptor:** the fixed pseudo-RGB input did as well as the learned adaptor.

### 07 Foundation models: DOFA and DINOv2 (`07_foundation_models.py`)

- **Input:** the same 128 × 128 leaf images as 06, standardised per band and upsampled to 224 × 224 (the size both models were pretrained at).
- **Training:** 20 epochs, batch 32, bf16, same augmentation as 06, no test-time augmentation. A new linear output layer (learning rate 1e-3) is always trained.
- **DOFA:** ViT-B/16, masked-autoencoder-pretrained on satellite data including hyperspectral EnMAP scenes (TorchGeo weights).
  - Its first layer builds its weights from each band's wavelength.
  - All 204 bands go in, with their wavelengths in µm; no adaptor is needed.
- **DINOv2:** ViT-S/14, pretrained on 142 million ordinary photos (LVD-142M). It uses the same 204 → 3 pseudo-RGB-initialised adaptor as 06, trained at learning rate 1e-3.
- **Two ways of using each:**
  - `probe`: the whole backbone frozen; only the output layer (and, for DINOv2, the adaptor) is trained.
  - `last2`: also unfreezes the last 2 transformer blocks and the final norm, at learning rate 5e-5.

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 07_dofa_probe | 0.550 / 0.531 / 0.491 | 0.644 / 0.638 |
| 07_dofa_last2 | 0.549 / 0.565 / 0.471 | 0.681 / 0.662 |
| 07_dinov2_probe | 0.587 / 0.517 / 0.542 | 0.657 / 0.644 |
| 07_dinov2_last2 | 0.583 / 0.542 / 0.544 | 0.664 / 0.622 |

How it did:
- **DINOv2 probe:** the highest within-variety score of the whole study for any disease (0.587). Compared with the best classical run it is not significantly better (+0.006, CI −0.046 to +0.073), and that pair was chosen after seeing the results.
- **Frozen DOFA:** training loss stayed near 0.63, close to chance, and it had the lowest pooled AUC of the deep models (−0.107 vs the pretrained ResNet).
- **DINOv2 with two blocks tuned:** reached training loss 0.33 but scored the same on held-out leaves, i.e. it overfitted.

### 08 Self-supervised pretraining on our own spectra (`08_self_supervised_pretraining.py`)

- **Step 1, pretraining without labels.**
  - Data: 500 random pixels from each training leaf (~400,000 pixels), standardised per band.
  - Task: about half the bands are hidden in contiguous blocks of 6 bands (~18 nm), and the network reconstructs them from the visible bands.
  - Encoder 204 → 256 → 128 (LayerNorm, GELU); decoder 128 → 256 → 204. The loss is mean squared error on the hidden bands only.
  - 20 epochs, batch 1024, learning rate 1e-3. Only training-leaf pixels are used; no labels and no test pixels.
- **Step 2, fine-tuning with labels.**
  - The decoder is dropped; the encoder gets a head (dropout 0.3, 128 → 64 → 2).
  - Trained exactly like 04: 100 pixels per leaf per epoch, same augmentation, 15 epochs, batch 512. Learning rate 3e-4 for the encoder (to keep what it learned) and 1e-3 for the head.
  - Test leaf = average over every pixel.
- **The two runs:**
  - `08_ssl_pretrained`: encoder from step 1.
  - `08_ssl_scratch`: the identical network from random weights (the control).

| run | within-variety AUC any / RB / LR | balanced acc. RB / LR |
|---|---|---|
| 08_ssl_pretrained | 0.543 / 0.575 / 0.477 | 0.703 / 0.670 |
| 08_ssl_scratch | 0.557 / 0.576 / 0.498 | 0.707 / 0.671 |

How it did: pretraining had no effect (−0.014, CI −0.034 to +0.007). The hidden bands were almost perfectly
predictable from their neighbours (reconstruction error 0.0097 in standardised units, where 1.0 = no better than the
mean). Leaf spectra are smooth enough that the task taught interpolation, not disease.

## Overall

- Red blotch has a small, real signal (within-variety AUC about 0.55–0.59; 27 of 47 runs had a 95% interval above 0.5).
- Leafroll shows none (0 of 47).
- All approaches are statistically tied.
- Much of any pooled score is the variety.

Full tables with intervals and average precision: `experiments/results/summary.md`, `experiments/results/paired_tests.md`.
Method details and caveats: `experiments/README.md`. Written analysis: `reports/Experiment results and analysis.md`.

## What is here

| Path | Contents |
|---|---|
| `experiments/01_…10_*.py`, `run_all.sh`, `third_party/dofa.py` | the scripts of the study (09 = comparison table, 10 = diagnostics) |
| `experiments/results/` | one out-of-fold prediction file per run, `summary.md/.csv`, `paired_tests.md`, `per_variety.csv`, diagnostics; `quick/` = smoke-test runs |
| `experiments/logs/` | the run logs |
| `reports/` | the literature report that chose the approaches, and the results analysis |

**Not moved** (now in `claude runs/shared/`, formerly `claude runs/experiments/`, because the new pipelines use them): `common.py`, `torch_common.py`,
`00_build_cache.py` (with its log) and the data cache `shared/cache/`. The literature notes in
`Classification/research_notes/` were also left in place.

To rerun an archived script, copy it next to `common.py` in `claude runs/shared/`. In the data cache,
`images.npy` and `masks.npy` (6.3 GB) are used only by scripts 06/07, and `leaf_stats.npy` only by 02/10. They can be
deleted if space is needed: `00_build_cache.py` rebuilds the first two, and script 02 rebuilds `leaf_stats.npy`.
