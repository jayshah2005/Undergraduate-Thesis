# Grapevine disease from hyperspectral leaves: what worked, what didn't, and why

Companion to `Hyperspectral grapevine transfer learning.md` (the literature report). That report said what to try; this one says what happened when we tried it on your 998 leaves. All numbers come from `experiments/results/` (47 runs, one full pass of `experiments/run_all.sh`, 45 min). Statements are marked **[measured]** when a result in this study shows them and **[hypothesis]** when they are my explanation and were not tested.

## 1. Bottom line

1. **The data contains only a weak disease signal, and no approach extracts much of it.** Compared inside a single variety, the best runs separate sick from healthy leaves with an AUC of about 0.55–0.59 (0.5 is chance, 1.0 is perfect). The spread between approaches is smaller than the statistical noise (±0.04–0.05). **[measured]**
2. **Red blotch has a small but real signal; leafroll shows none.** Of 47 runs, 27 have a within-variety red-blotch AUC whose 95% interval lies above 0.5. For leafroll the count is 0 of 47. **[measured]** Combined "any disease vs healthy": 15 of 47. About one run in 47 exceeding 0.5 by chance alone is expected; the runs are correlated, so treat "27 of 47" as "consistently a little above chance", not as 27 independent confirmations.
3. **Most of the apparent accuracy is the variety.** A "model" that looks only at the variety scores AUC 0.745 for red blotch and 0.616 for any disease, because 96% of Cab Franc leaves have red blotch while only 35% of Chardonnay leaves do. Every pooled number below about 0.75 is mostly this. **[measured]**
4. **Transfer learning did not give a reliable gain.** ImageNet-pretrained ResNet-18 beat the identical network from random weights on the pooled metric (+0.047, 95% CI +0.008 to +0.080) but not within varieties (+0.021, CI −0.027 to +0.064). Self-supervised pretraining and the hyperspectral foundation model (DOFA) gave no gain either. **[measured]**
5. **Models do not carry over between varieties.** Trained on three varieties and tested on the fourth, within-variety AUC is 0.47–0.55 (chance), and pooled leafroll AUC falls to 0.29–0.45 (below chance). The spectral signal points in different directions in different varieties. **[measured]**

What this means in practice: with these labels and this imaging, a spectral classifier will rank leaves only slightly better than chance once variety is accounted for. The choice of model (classical, CNN, pretrained or not) matters much less than the limited signal in the data.

## 2. What was run

- **Data:** 998 single-leaf cut-outs, 204 bands (400–1000 nm), 4 varieties. Two labels per leaf: red blotch (RB) and leafroll 3 (LR). Healthy = both negative (226 leaves, 23%).
- **Validation:** 5 folds where fold k = every leaf of row Rk in all four varieties. Whole rows are held out together, pixels never cross the split, scalers are fitted on training leaves only, and nothing is tuned on test folds. Cab Franc is left out of leafroll (1 positive in 250 leaves).
- **Two kinds of AUC.** *Pooled* AUC mixes all varieties. *Within-variety* AUC only compares leaves of the same variety and fold, so a model that merely recognises the variety scores exactly 0.5. The within-variety number answers the real question.
- **Approaches**, matching the five options in the literature report plus controls:

| script | approach |
|---|---|
| 01 | variety-only baseline (a bar, not a model) |
| 02 | SVM / logistic / PLS-DA / boosting on one summary spectrum per leaf: 4 feature sets × 4 models, also one model per variety and leave-one-variety-out |
| 03 | the same idea on individual pixels, averaged per leaf |
| 04 | pixel MLP and 1D CNN |
| 05 | attention multiple-instance learning (a leaf is a bag of pixels) |
| 06 | ResNet-18 on whole leaf images: ImageNet-pretrained vs random weights, plus a fixed pseudo-RGB control |
| 07 | DOFA (hyperspectral foundation model) and DINOv2 (RGB model with a band adaptor), frozen and last-2-blocks tuned |
| 08 | self-supervised masked-band pretraining vs the same network from scratch |

## 3. Results

### 3.1 By approach family (within-variety AUC, any disease; 0.5 = chance)

| family | settings | worst / median / best |
|---|---|---|
| 07 foundation models | 4 | 0.549 / 0.566 / **0.587** (`07_dinov2_probe`) |
| 02 classical, one model per variety | 16 | 0.475 / 0.540 / 0.581 |
| 06 ResNet-18 (3 runs) | 3 | 0.553 / 0.561 / 0.574 |
| 02 classical, one pooled model | 16 | 0.483 / 0.532 / 0.558 |
| 08 self-supervised | 2 | 0.543 / 0.550 / 0.557 |
| 03 classical on pixels | 2 | 0.524 / 0.533 / 0.542 |
| 04 pixel MLP / 1D CNN | 2 | 0.522 / 0.530 / 0.538 |
| 05 attention MIL | 1 | 0.527 |
| 01 variety prior | 1 | 0.500 |

The "best" in each row is the best of several settings tried without tuning, so it is optimistic. Individual runs carry roughly ±0.05 (95% bootstrap), so nearly all of these are statistically indistinguishable from each other. The best-looking family (foundation models) is chosen after the fact and is not significantly better than the best classical run (difference +0.006, CI −0.046 to +0.073).

### 3.2 Per label (best within-variety AUC across all runs)

| label | best | typical run | runs with 95% interval above 0.5 |
|---|---|---|---|
| red blotch | 0.589 (`02_logreg__mean_raw__byvariety`) | 0.54–0.57 | 27 of 47 |
| leafroll | 0.544 (`07_dinov2_last2`) | 0.47–0.52 | 0 of 47 |
| any disease (healthy detection) | 0.587 | 0.52–0.56 | 15 of 47 |

### 3.3 Head-to-head questions (paired bootstrap on the same leaves, 95% CI of the difference)

| question | pooled AUC, any | within-variety AUC, any | verdict |
|---|---|---|---|
| ImageNet-pretrained ResNet vs same net from scratch | +0.047 [+0.008, +0.080] | +0.021 [−0.027, +0.064] | pooled gain only |
| Self-supervised pretraining vs scratch (same net) | +0.000 [−0.011, +0.011] | −0.014 [−0.034, +0.007] | no effect |
| Frozen DOFA vs pretrained ResNet | −0.107 [−0.155, −0.063] | −0.024 [−0.076, +0.038] | DOFA worse pooled |
| Attention-MIL vs plain pixel averaging | +0.018 [−0.008, +0.049] | +0.005 [−0.042, +0.058] | no difference |
| Pretrained ResNet vs pixel 1D CNN from scratch | +0.053 [+0.008, +0.097] | +0.036 [−0.019, +0.091] | pooled gain only |
| Best run (DINOv2 probe) vs best classical (post-hoc pair) | +0.003 [−0.034, +0.043] | +0.006 [−0.046, +0.073] | no difference |

Control: the ResNet with a fixed pseudo-RGB input (no learned band adaptor) scored 0.561 within variety and 0.691 pooled, against 0.574 and 0.696 with the learned adaptor. Learning the 204→3 band combination added nothing measurable.

### 3.4 Where the signal is, by variety (within-variety AUC; selected runs)

| run | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| red blotch: `04_pixel_cnn1d` | 0.56 | 0.54 | 0.62 | 0.57 |
| red blotch: `02_svm__stats_raw` | 0.50 | 0.61 | 0.60 | 0.61 |
| red blotch: `06_resnet18_pretrained` | 0.47 | 0.52 | 0.57 | 0.58 |
| leafroll: `07_dinov2_probe` | 0.61 | n/a | 0.54 | 0.39 |
| leafroll: `04_pixel_cnn1d` | 0.45 | n/a | 0.53 | 0.58 |

Leafroll AUCs swing from 0.39 to 0.61 for the same variety depending on the model. That is the signature of noise: Vidal has only 25 leafroll-positive leaves in total.

### 3.5 Leave-one-variety-out stress test

The best of 16 runs reaches within-variety AUC 0.553 for any disease; most are 0.47–0.55. Pooled red-blotch AUC is 0.38–0.52 and pooled leafroll AUC 0.29–0.45. A model trained on the other three varieties cannot rank leaves in the held-out variety.

### 3.6 Average precision (healthy-leaf detection)

Healthy leaves are 22.6% of the data. Counting "healthy" as the positive class, the best runs reach average precision 0.31–0.36 against a chance level of 0.226, i.e. a gain of +0.10 to +0.13 within varieties (`07_dinov2_probe` +0.133, `02_svm__stats_raw` +0.126, `06_resnet18_pretrained` +0.126). Practically: the models rank healthy leaves somewhat earlier than a random ordering would, but far from cleanly (average precision is not the same as the share of correct top picks, so read it only as 'above chance by about 0.1').

## 4. Why things did or did not work

### 4.1 Why almost nothing worked: the signal is small and varies by variety

- **Single bands carry little.** Within a variety, the best single band reaches AUC 0.536 for red blotch (704–710 nm and 557 nm) and 0.531 for leafroll (536–545 nm). Averaged over varieties, the curves stay within 0.50–0.54 (`results/diagnostics_band_auc.png`). **[measured]** The locations match the literature (green/red-edge region), which is reassuring: this is the real biology, only small.
- **The sign of the effect depends on the variety.** For red blotch, reflectance at 530–620 nm and ~705 nm separates sick from healthy in Chardonnay, Vidal and Cab Franc, but the relationship is reversed in Baco (AUC below 0.5 in the same bands). **[measured]** A pooled model must learn "if Chardonnay, higher means sick; if Baco, higher means healthy", which is hard with a small, noisy signal. This explains why leave-one-variety-out collapses and why per-variety models help a little (best 0.581 vs 0.558 pooled, not significant).
- **Why the signal may be small [hypothesis, untested]:** infected leaves may not show symptoms yet at capture (August 12); a single leaf's mean spectrum averages over patchy symptoms; the labels come from a different test (PCR or scouting) than the imaging. The literature report found the same range (80–87% binary accuracy in the closest grapevine study, with likely leakage).
- **Leakage is not hiding extra signal.** Random splits scored the same as row-grouped splits in our check (e.g. red blotch within-variety AUC 0.525 vs 0.527). **[measured]** Neighbouring vines do share labels (leafroll: AUC 0.72 predicting a vine from its two neighbours), so row-grouping is correct, but these models found too little signal to exploit it.

### 4.2 Classical models on leaf-mean spectra (script 02)

- **What happened:** PLS-DA, logistic regression and SVM were mutually indistinguishable (best 0.558 within variety). Boosting was worst: `02_hgb__mean_raw` scored 0.483 and its per-variety version 0.475, below chance. **[measured]**
- **Preprocessing (SNV, derivative, extra statistics) had no consistent winner. [measured]** Both match the literature report.
- **Why [hypothesis]:** these models fit smooth functions of correlated bands, which suits a weak, smooth signal. Trees split on individual bands among 204–816 near-duplicate features with ~800 training leaves, so they chase noise.
- **Per-variety models** were the best classical option because they remove the sign problem, at the cost of 4× less data each. Cab Franc cannot be modelled this way (9 healthy leaves).

### 4.3 Pixel-level models (scripts 03, 04) and attention MIL (05)

- **What happened:** pixel logistic/boosting, MLP and 1D CNN all landed at 0.52–0.54 within variety, no better than leaf-mean classical models. Attention-MIL was no better than averaging (+0.005, CI −0.042 to +0.058). **[measured]**
- **They barely fit even the training data:** final training loss was 0.57–0.60, barely below the 0.64 of the first epoch. **[measured]** Many pixels in a diseased leaf look just like healthy-leaf pixels, so a pixel's label is mostly noise. Averaging over the leaf helps, but nothing here finds the diseased patches.
- **Why MIL did not help [hypothesis]:** MIL should help only if some pixels are clearly diseased. The training loss says the model never found such pixels, which fits a weak or absent local symptom at capture.
- **Using all pixels did not add information.** Pixels in a leaf are strongly correlated, so millions of pixels behave like ~800 independent examples.

### 4.4 Transfer learning with ImageNet ResNet-18 (script 06): the one place with a visible effect, but a shaky one

- **What happened:** pretrained beat scratch by +0.047 pooled (CI excludes zero), but by +0.021 within varieties (CI includes zero). **[measured]** The pretrained net also reached much lower training loss (0.20 vs 0.375 at the last epoch, fold 0). **[measured]** So it fits the training leaves much more easily, but this does not turn into better ranking on held-out leaves.
- **Where the pooled gain likely comes from [hypothesis]:** ImageNet features pick up coarse colour and texture that distinguish varieties, which raises the pooled score. The within-variety score, which excludes that, does not significantly improve.
- **The learned adaptor was not needed:** a fixed pseudo-RGB input (0.561) did as well as the learned 204→3 layer (0.574). **[measured]** ImageNet-style features appear to use the coarse leaf appearance, not fine spectral detail. **[hypothesis]**
- **This agrees with the literature report**, which rated this the best-supported transfer option, but noted that its evidence came from satellite images and that the benefit shrinks as data grows. Here the benefit shrinks to zero once variety is removed.

### 4.5 Foundation models (script 07)

- **DOFA (hyperspectral, frozen):** worst pooled score of the deep models (0.588; −0.107 vs the pretrained ResNet) and training loss stuck at 0.63, close to chance. **[measured]** Tuning its last two blocks recovered some of it (0.665 pooled). **[measured]** **Why [hypothesis]:** it was pretrained on satellite and airborne pixels (tens of metres per pixel), not on single leaves; the leaf spectra may sit outside what its frozen features encode. The literature report predicted exactly this and ranked it fourth.
- **DINOv2 with a band adaptor:** the top-ranked run by within-variety score (probe 0.587, last-2 tuned 0.583), but its advantage over classical models is not significant (+0.006, CI −0.046 to +0.073), and the pair was chosen after the results were seen. DINOv2 with all but the head frozen cannot be memorising much (training loss 0.58), which makes it a cautious, low-variance model: a plausible reason it did relatively well in a noisy problem. **[hypothesis]**
- **Overfitting when more of the model is trainable:** DINOv2 with two blocks tuned reached training loss 0.33 yet scored the same on held-out leaves. **[measured]**

### 4.6 Self-supervised pretraining (script 08)

- **What happened:** pretrained vs scratch: −0.014 (CI −0.034 to +0.007). No effect. **[measured]**
- **Why [hypothesis]:** the masked-band reconstruction error fell to 0.0097 in standardised units (1.0 = no better than predicting the mean), i.e. hidden bands are almost perfectly predictable from visible neighbours. Leaf spectra are so smooth and redundant that this task teaches smooth interpolation, not disease. The literature report expected a null result as often as a gain.

### 4.7 Leafroll specifically

No run beat chance on leafroll within varieties (0 of 47). Possible reasons **[hypothesis]**: leafroll often appears as downward-rolled red or yellow leaves later in the season than the imaging date; Cab Franc has no positives and Vidal only 25, leaving little to learn from; and leafroll labels cluster strongly by row (neighbour AUC 0.72), so apparent signal can easily be a row effect that grouped validation removes.

### 4.8 Why ROC AUC, and whether average precision or something else would be better

- **AUC is the right main metric here** because the label balance differs hugely between varieties and folds (red blotch 35% to 96%). AUC does not depend on prevalence and does not need a threshold, so it lets us compare across them. We use it stratified (within-variety) so variety cannot inflate it.
- **Average precision is the right complement for healthy-leaf detection**, since healthy is the rare class (23%) and AP focuses on how clean the top-ranked picks are. Its weakness is that the chance level equals the prevalence, which differs per variety, so we report "AP gain over chance". It tells the same story as AUC (best gain +0.133; ranking of runs largely similar), so the conclusions do not depend on this choice.
- **Not used on purpose:** accuracy and balanced accuracy at 0.5 (they depend on a threshold; `bal.acc ANY` is ≈0.5 for most runs because two class-weighted heads push almost every "any disease" probability above 0.5, an artefact rather than a bug) and calibration scores like Brier (the class-weighted losses distort probabilities). If you move to a decision rule (e.g. "flag the 20% most-likely healthy vines"), report precision and recall at that operating point.

## 5. Which findings to trust

- **Strong:** variety explains much of the pooled accuracy; models do not generalise across varieties; no leafroll signal within varieties; pretraining effects are small or zero within varieties. These hold across many runs, with intervals.
- **Moderate:** red blotch has a weak real signal (27/47 runs above 0.5, best ~0.59).
- **Weak:** which single run or family is "best". Differences are within noise, and several picks were made after seeing results. Do not choose a model from the leaderboard order.
- **Not tested:** hyperparameter tuning (everything used fixed defaults), multiple random seeds for neural networks, different image sizes, other backbones, and any explanation in this report marked as a hypothesis.

## 6. What I would try next (suggestions, untested)

1. **Check the labels and timing.** Ask when each leaf was imaged relative to symptom onset and how the labels were obtained. If many positives are asymptomatic, no leaf-level spectral model will separate them well.
2. **Build the problem around variety.** Train one model per variety with strong regularisation, or add the variety as an explicit input with interactions, and collect more healthy Cab Franc leaves (9 now) and more leafroll-positive Vidal (25).
3. **Look for row and imaging effects** (lighting, time of day). They can create apparent signal within a row that does not generalise.
4. **Get more independent leaves rather than a bigger model.** The deep models overfit within a few epochs; the gain from 1M pixels is small because the information is per leaf.
5. **Use several leaves per vine, or higher-resolution images** to find patchy symptoms (the diseased region, not the whole-leaf average).
6. **Repeat the best few settings over several random seeds** before relying on any ranking.

## 7. Reproducibility and timing

`./experiments/run_all.sh` ran end to end in **45 min 25 s** on 14 CPU cores and an RTX PRO 5000: 01 <1 s; 02 pooled 1 m 50 s; 02 per-variety 7 s; 02 leave-one-variety-out 1 m 43 s; 03 2 m 25 s; 04 2 m 38 s; 05 9 m 32 s; 06 7 m 7 s; 06 fixed-RGB control 3 m 36 s; 07 12 m 56 s; 08 3 m 10 s; diagnostics 2 s; comparison 19 s. A re-run reproduced earlier numbers (two logistic-regression probability sets differ by ≤0.008).

Files: scripts and `README.md` in `experiments/`; every run's out-of-fold probabilities in `experiments/results/*_oof.csv`; tables in `results/summary.md` and `results/paired_tests.md`; figure `results/diagnostics_band_auc.png` (the leafroll panel's curves for Vidal and Chardonnay were mislabeled in an earlier version of this figure and are fixed).
