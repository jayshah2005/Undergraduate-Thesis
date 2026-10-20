# Grapevine disease classification from hyperspectral leaf cubes — experiments

One script per approach from the research report
(`../reports/Hyperspectral grapevine transfer learning.md`), all scored the same way.

**Task:** 998 leaf cut-outs (Specim IQ, 204 bands, 400–1000 nm, 4 varieties). Two yes/no labels per leaf:
`red_blotch` and `leafroll3`. *Healthy* = both negative; *any disease* = either positive.

## 1. Start here: what the results say

**Detailed write-up of results and the reasons behind them: `../reports/Experiment results and analysis.md`.**
Full tables: `results/summary.md` (all 47 runs + leave-one-variety-out + average precision) and `results/paired_tests.md`.
All numbers are AUC (0.5 = coin flip, 1.0 = perfect). The "within-variety" column only compares leaves of the **same variety**, so it cannot be inflated by knowing the variety.

| approach family | settings tried | within-variety AUC, any disease: worst / median / best |
|---|---|---|
| 07 foundation models (DOFA, DINOv2) | 4 | 0.549 / 0.566 / **0.587** (`07_dinov2_probe`) |
| 02 classical on leaf-mean spectra, one model per variety | 16 | 0.475 / 0.540 / 0.581 |
| 06 ResNet-18 (pretrained, scratch, fixed-RGB control) | 3 | 0.553 / 0.561 / 0.574 |
| 02 classical on leaf-mean spectra (one pooled model) | 16 | 0.483 / 0.532 / 0.558 |
| 08 self-supervised pretraining | 2 | 0.543 / 0.550 / 0.557 |
| 03 classical on pixels | 2 | 0.524 / 0.533 / 0.542 |
| 04 pixel MLP / 1D CNN | 2 | 0.522 / 0.530 / 0.538 |
| 05 attention MIL | 1 | 0.527 |
| 01 variety prior (looks only at the variety, not a model) | 1 | 0.500 |

**Findings (read these with the caveats in section 5):**

1. **The spectral signal is weak. No approach is clearly better than the others.** Within a variety the best runs reach about 0.55–0.59 AUC for "any disease". Differences between families are inside the noise (a 95% bootstrap interval on a paired difference is about ±0.05).
   - **Red blotch: a small real signal.** 27 of 47 runs have a within-variety 95% interval entirely above 0.5 (best ≈ 0.59).
   - **Leafroll: none detected.** 0 of 47 runs. "Any disease": 15 of 47. (Runs are correlated and many were screened, so read this as "consistently a little above chance" for red blotch, not 27 independent confirmations.)
2. **Much of the headline accuracy is the variety.** Guessing from the variety alone gives pooled AUC 0.745 (red blotch) and 0.616 (any disease): 96% of Cab Franc leaves have red blotch, 35% of Chardonnay leaves do. Pooled numbers above ~0.7 are mostly this.
3. **Is transfer learning worth it? Not clearly, here.**
   - ImageNet-pretrained ResNet-18 vs the same net from random weights: pooled AUC for "any disease" +0.047 (95% CI +0.008 to +0.080), but within-variety only +0.021 (CI −0.027 to +0.064), i.e. not distinguishable from zero.
   - Self-supervised pretraining on our own spectra vs the same net from scratch: no effect (−0.014, CI −0.034 to +0.007).
   - Frozen hyperspectral foundation model (DOFA) vs ImageNet ResNet: no difference within variety (−0.024, CI −0.076 to +0.038); lower pooled (−0.107), meaning it captures less variety information.
   - A fixed pseudo-RGB input (no learned band adaptor) did as well as the learned 204→3 adaptor (0.561 vs 0.574 within variety).
   - The best single run (DINOv2 with a band adaptor, frozen backbone, 0.587) is not significantly better than the best classical run (+0.006, CI −0.046 to +0.073; a pair chosen after seeing the results).
4. **Signal direction differs between varieties** (`results/diagnostics_band_auc.png`). For red blotch, reflectance at 530–620 nm and around the red edge (~705 nm) separates sick from healthy leaves in Chardonnay, Vidal and Cab Franc, and the effect is **reversed in Baco**. A single pooled model cannot use a signal whose sign depends on the variety.
   - Leave-one-variety-out (the model never saw the test variety): within-variety AUC ≈ 0.47–0.55, and pooled leafroll AUC 0.29–0.45 (*below* chance because the direction flips). Models do not transfer across varieties.
   - One model per variety helps a little for classical models (best 0.581 vs 0.558 pooled), but not significantly. Chardonnay and Vidal carry the most signal (red blotch ~0.55–0.62 within variety); Baco carries almost none.
5. **Neighbouring vines have similar labels** (leafroll especially: AUC 0.72 from the two neighbours' labels), which is why folds are whole rows. In our checks random splits did *not* inflate the scores (`results/diagnostics.txt`, section C), because these models find little signal to leak.

Possible reasons for the weak signal (not tested here): many infected leaves may have no visible symptoms yet at capture time; single-leaf cut-outs average over patchy symptoms; labels come from a different test than the imaging.

## 2. How to run

```bash
cd /home/jay/Desktop/CODE/Classification
./experiments/run_all.sh quick     # ~3 min smoke test, writes results/quick/
./experiments/run_all.sh           # full study, 45 min measured, writes results/summary.md
```

Or run any step alone (`.venv/bin/python experiments/0X_name.py`; most take `--quick`).
The first step (`00_build_cache.py`, ~1 min) reads the 4 GB HDF5 once and writes `experiments/cache/` (~13 GB of `.npy`).

| file | what it does | report option | time |
|---|---|---|---|
| `common.py` | paths, folds, metrics — **the evaluation rules, read this first** | — | — |
| `torch_common.py` | the PyTorch training loop with every step explained | — | — |
| `00_build_cache.py` | HDF5 → fast `.npy` caches; checks that zeros are whole pixels | — | 1 min |
| `01_baseline_variety_prior.py` | "guess from the variety" — the bar a real model must beat | validation | secs |
| `02_classical_leaf_level.py` | SVM / logistic / PLS-DA / boosting on one summary spectrum per leaf (`--by-variety`, `--lovo`) | #1 | ~2 min (16 settings) |
| `03_pixel_level_classical.py` | same idea on individual pixels, averaged per leaf | #1 | ~2.5 min |
| `04_pixel_mlp_cnn1d.py` | small neural nets on pixel spectra | #2 | 2.5 min |
| `05_attention_mil.py` | leaf = bag of pixels, attention picks the informative ones | #2 (variant) | 9.5 min |
| `06_resnet_transfer.py` | ResNet-18 on whole leaf, **pretrained vs scratch** with a learned 204→3 band adaptor (`--adaptor fixed_rgb` = pseudo-RGB control) | #3 | 7 min (+3.5 min control) |
| `07_foundation_models.py` | DOFA (hyperspectral) and DINOv2 (RGB + adaptor), frozen and last-2-blocks tuned | #4 | 13 min |
| `08_self_supervised_pretraining.py` | masked-band autoencoder pretraining, then fine-tune; vs scratch | #5 | ~3 min |
| `09_compare.py` | one table for everything + average precision + paired bootstrap tests | — | 19 s |
| `10_diagnostics.py` | where is the signal? spatial clustering, leakage check, per-band AUC | — | 2 s |
| `third_party/dofa.py` | DOFA model code from TorchGeo (MIT licence), unchanged | — | — |

**New to PyTorch?** Read in this order: `torch_common.py` (concepts + training loop) → `04_pixel_mlp_cnn1d.py` (a small model) → `06_resnet_transfer.py` (transfer learning). Every model/dataset/training step is commented.

## 3. The evaluation rules (identical for every script)

- 5 folds; fold *k* = all leaves in row R*k* of every variety. Whole rows are held out together (neighbours look alike).
- Pixels never cross the split. Scalers and standardisation are fitted on training leaves only. No tuning or early stopping on the test fold; fixed epochs and default settings.
- Cab Franc has 1 leafroll-positive leaf, so it is excluded from leafroll training and scoring (it is still used for red blotch).
- Each script saves out-of-fold probabilities per leaf to `results/<name>_oof.csv`; `09_compare.py` scores them all in one way.
- **Metrics.** ROC AUC is the main metric (independent of how common each label is, which differs hugely between varieties). Average precision is reported too, for healthy-leaf detection, as "AP gain over chance". `bal.acc ANY` in the tables is ≈0.5 for most runs; that is a threshold artefact (two class-weighted heads push almost every P(any disease) above 0.5), not a bug.
- "Within-variety" AUC = stratified AUC (compare only leaves of the same variety and fold).
- Plus/minus values are 95% bootstrap half-widths over leaves.

## 4. Which ideas were left out, and why

From the report's "deprioritized" list: SpectralGPT / HyperSIGMA (need 12 bands or a re-initialised input layer, which discards the pretraining), HyperFree and the proximal-sensing MAE (weights not available), SpectralEarth / Copernicus-FM (satellite scale), pretraining on Indian Pines / Pavia (different sensor and scale), heavy 3D-CNNs, generative augmentation. DOFA (#4) is the one foundation model that accepts all 204 bands with wavelengths, so it is the one tested.

## 5. Caveats — please read before trusting any number

- **No hyper-parameter tuning.** Defaults were fixed in advance. Script 02 has 16 settings per model family, so its "best" number is optimistic (picked from 16); the family table shows worst/median/best for that reason.
- **Noise.** With ~1000 leaves and weak signal, differences below ~0.05 AUC should be treated as ties. Neural-network runs use one seed; seed-to-seed variation was not measured.
- **Within-variety numbers have no interval in the main table**; the paired tests (`results/paired_tests.md`) give intervals for the five comparisons that matter.
- DOFA/DINOv2 settings (20 epochs, learning rates 1e-3 head / 5e-5 blocks) were not tuned. A different recipe could change them.
- Two settings of script 02/03 changed by ≤0.008 in a few probabilities between runs (solver numerics after parallelising); conclusions are unaffected.
- Several literature claims in the report come from search summaries rather than full papers (flagged there).
- The first version of `results/diagnostics_band_auc.png` had mismatched colours in the leafroll panel (fixed): the strong blue-band curve there is Vidal, which has only 25 leafroll-positive leaves, so it should not be trusted.

## 6. Speed notes (14 cores, RTX PRO 5000 48 GB, 62 GB RAM)

- CPU scripts (02, 03) run folds/settings in parallel processes (`--jobs`, default = half the cores) with 2 threads each; running them alongside GPU jobs causes thread thrashing, so `run_all.sh` runs steps one after another.
- Optimised: 02 and 03 (parallel folds/settings; ≈ 2 min and 2.5 min, previously tens of minutes because overlapping jobs oversubscribed CPU threads) and 09 (next line). The GPU scripts' training loops were left alone because they were not the bottleneck: 06/07 take ≈ 40 s per fold, and 07 is the longest step (13 min).
- `09_compare.py` was slow (> 7 min) because the bootstrap recomputed the full scikit-learn metric set 200 times for each run on one core. It now uses a rank-based AUC, a lean bootstrap and parallel processes: 19 s.
- Caches are `.npy` files opened with memory-mapping, so all processes share one copy in RAM (the cache folder uses 13 GB of disk).
- Added dependency: `timm` (in `../requirements.txt`).
