# Analysis plan (fixed 2026-11-25, before the runs)

Everything here was decided before any of the runs it covers. Results go on one page, `RESULTS.md`, built by one
script from the saved predictions.

## Data, the same for every model

- 998 leaves, 204 bands or pseudo-RGB (the bands nearest 640 / 550 / 460 nm), labels = FINAL 2026 virus tests.
- Each pixel is normalized over its own bands by `helpers/cnn_common.normalize_pixels`: SNV over 204 bands; normalized
  rgb (each band / R+G+B; Gevers & Smeulders 1999) over the 3 RGB bands. The CNNs see these pixels on a 212 px canvas
  (stored in float16); logistic regression sees each leaf's average of the same pixels.
- Folds: test row Rk, validation row R(k+1), train on the other 3 (of every variety in a pool).
- CNNs: 5 folds × 5 seeds (seed = 100 · s + fold), one prediction per leaf = the mean probability of the 5 seeds.
  Logistic regression: 5 folds, one fit each.

## Models

| category | data | models |
|---|---|---|
| Variety (4 classes) | all 998 | LR; small CNN and small ResNet, each plain / + ECA / + CBAM × mean / attention pooling (12 CNNs) |
| Disease (leafroll, red blotch, separately) | within Baco, Chardonnay, Vidal; pooled all 998; pooled without Cab Franc (Baco + Chardonnay + Vidal) | the same 13 models |
| MOO (six heads: 4 varieties, red blotch, leafroll) | all 998 | small CNN and small ResNet (plain, attention pooling) × AdamW and SGD × LS, MGDA, MGDA-GN, -LN, -LGN, SVD-PGA, OSNMF-PGA, UPGrad |

Every model runs on 204 bands and on RGB. Both networks use two 2×2 max pools. Leafroll never includes Cab Franc (it
has 1 leafroll leaf); "pooled without Cab Franc" trains without it. Red blotch on all 998 is also scored without Cab
Franc's leaves ("scored without Cab Franc", same trained models), since 241 of its 250 leaves have red blotch.

Training (variety, disease and MOO with AdamW): batch 8, AdamW lr 1e-3 and weight decay 1e-2, dropout 0.5, random 90°
rotations and flips, up to 80 epochs, early stopping on the validation row's loss with patience 15. Disease and MOO
losses weight each class equally. MOO keeps each task's own best validation epoch (variety: the 4 variety heads'
summed loss); a one-checkpoint score (all six heads' summed loss) is reported beside it. Every aggregator computes in
float64.

Logistic regression (both categories): StandardScaler and class-balanced logistic regression on the 3 training rows;
C from {1e-5, …, 1e3} chosen by the validation row's class-balanced log-loss.

## Scores

- **Balanced accuracy** (main): the mean recall over classes, a leaf called positive (disease) at P ≥ 0.5 or its most
  likely variety. Chance 0.5 for disease, 0.25 for variety.
- **AUC**: threshold-free; for variety the mean of the four one-vs-rest AUCs.
- 95% intervals from 2000 resamples of leaves within each row × class cell (× variety for a pool), seeded, so every
  model scored on the same leaves gets the same resamples.

## Comparisons, and when a difference counts

Every comparison is shown in one table with both models' scores and the paired difference A − B with its 95%
interval, for balanced accuracy and for AUC. A difference is marked ✓ (A better) or ✗ (A worse) only if **both**:

1. the paired 95% interval of A − B excludes 0, and
2. A beats B (or loses to it) in at least 4 of the 5 seeds trained separately (seed s of A against seed s of B; a
   CNN's seed against logistic regression's single fit).

Unmarked rows are ties within the noise and stay in the table. Each table states how many rows were marked against the
1 in 20 expected by chance per score.

The comparisons:

1. Each CNN against logistic regression, same data.
2. ECA and CBAM against the same network and pooling without them; attention against mean pooling.
3. Small ResNet against small CNN, same module and pooling.
4. 204 bands against RGB, same model.
5. MOO: each aggregator against LS, same network, optimizer and bands; SGD against AdamW, same method.
6. Six-head LS (AdamW) against the single-task model with the same network and attention pooling, for variety,
   leafroll and red blotch.

**Reference for pooled disease sets: the variety-only rule.** For each fold, each variety's disease rate on the 3
training rows, turned into P with the same class weighting as the losses; it uses no spectrum. A pooled model has
to beat it to show more than the varieties' different disease rates. Within one variety it is exactly chance.

## SGD (MOO)

Momentum 0.9, weight decay 5e-4, constant learning rate. The learning rate is chosen on the validation rows only
(seed 0, every fold). How it is chosen (one rate from LS for every method, or one per method) is decided before the
SGD runs and written here.
