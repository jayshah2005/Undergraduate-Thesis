# PMMFNet controlled experiments (plan.md): report

All numbers are leaf-level and out-of-fold, using the original 5 row folds (998 leaves, 4 classes).
The full tables are in `runs/comparison.md`, and per-experiment reports are in `runs/<exp>/<exp>_metrics.md`.

## Best configuration: S1 (pre-declared rule, weak evidence)

Configurations were ranked only by **mean best validation 4-class balanced accuracy**. Test results played no part in the choice.

| experiment | what changed | val bal. acc. (selection) | test bal. acc. | macro-F1 | mean OvR AUC | within-variety AUC | runtime |
|---|---|---|---|---|---|---|---|
| Original | historical run: min-val-loss checkpoint, scaler on train+val | (0.416)* | 0.398 | 0.373 | 0.683 | 0.514 | 35 min |
| A1 | checkpoint = max val bal. acc. | 0.439 | 0.413 | 0.385 | 0.698 | 0.512 | 42 min |
| A2 (= L1) | A1 + scaler on training leaves only | 0.436 | 0.402 | 0.378 | 0.685 | 0.522 | 39 min |
| **S1** | **A2, no spatial shuffle** | **0.443** | **0.437** | **0.417** | **0.715** | **0.530** | 44 min |
| S2 | A2, fixed shuffle also at val/test | 0.433 | 0.403 | 0.373 | 0.693 | 0.519 | 48 min |
| S3 | A2 settings rerun (shuffle in training only) | 0.433 | 0.406 | 0.375 | 0.691 | 0.509 | 41 min |
| L2 | A2 with leaf-level (MIL) loss | 0.435 | 0.411 | 0.393 | 0.690 | 0.519 | 45 min |

\* Original's value is the validation balanced accuracy at its minimum-loss epoch. It was not selected on this metric, so it is not comparable.

**S1 wins the selection, and it is also the best on all four summary test metrics above.** Its settings: scaler fitted on training leaves only, checkpoint by validation balanced accuracy, patch-level loss, and no spatial shuffle anywhere.

The margin is small, though:
- S1's validation lead (+0.004 over A1, +0.010 over S3) is the size of the spread between A1, A2 and S3 (0.433–0.439), which have identical settings in 3 of 5 folds.
- The test number comes from the same trained models whose validation scores picked it, so 0.437 is likely optimistic for the configuration.
- Confirming S1 needs at least 3 seeds per arm.

## What the experiments show

1. **No setting changes the main result: within a variety, the model is close to chance.**
   - Mean within-variety OvR AUC is 0.51–0.53 in every experiment.
   - Red blotch within-variety AUC is 0.56–0.58. Leafroll is 0.47–0.51.
   - The mean per-variety 4-class balanced accuracy is 0.27–0.31; chance is 0.25–0.33, depending on how many classes a variety has.
   - So the level above chance (0.40–0.44 pooled vs 0.27–0.31 per variety) is likely mostly variety recognition. The gap is about 0.12 in every experiment (0.110–0.132).
   - The differences between configurations sit inside the varieties. From Original to S1, pooled accuracy rises by 0.039, per-variety accuracy by 0.033 (0.274 → 0.307) and within-variety AUC by 0.016. These gains are small, and disease detection within a variety stays weak.
2. **Same seed, different run.** A fixed seed does not reproduce a run on this setup.
   - In fold 1, A1 and S3 match to within 1e-6, but A2 already differs there at epoch 1 (train loss 1.1898 vs 1.1901). The likely cause is `cudnn.benchmark=True` (inherited from the original), which picks GPU kernels by timing; this was not tested.
   - Where experiments share identical settings (folds 1–3: A1, A2, S3; folds 0 and 4: A2, S3), test balanced accuracy varies by up to **0.038** within a fold.
   - S3 vs A2 overall: +0.004.
3. **Checkpoint selection (A1 vs Original): +0.014**, better in 3 of 5 folds. The CI [−0.013, +0.041] includes 0. The selected epochs move later (mean 2.2 → 4.0).
4. **Scaler on training leaves only (A2 vs A1): −0.011, within noise.**
   - In folds 1–3, the validation row does not change the global min/max, so A1 and A2 are the same settings there.
   - That leaves only folds 0 and 4 to test the scaler.
   - The leak-free scaler costs nothing measurable, so keep it.
5. **Spatial shuffle: removing it (S1) gives the largest gain, but unevenly across folds.**
   - The training-shuffle configuration was run twice (S3 and A2). S1 beats S3 by +0.031 (bootstrap CI [+0.006, +0.055], 4/5 folds) and A2 by +0.035 (CI [+0.004, +0.064], 3/5 folds).
   - Against the widest rerun range (±0.038), S1 is clearly better only in fold 1 vs S3, and in folds 1–2 vs A2. The gain is concentrated in folds 1–2.
   - The bootstrap holds the trained models fixed, so it does not include training noise.
   - Adding the fixed shuffle at evaluation (S2) has no overall effect: −0.004 vs S3 and +0.001 vs A2.
6. **Leaf-level MIL loss (L2): +0.009 vs A2 and +0.005 vs S3.**
   - It is the most fold-consistent change: higher than A2 in 5/5 folds (fold 0 by only 0.001) and higher than S3 in 4/5 (lower in fold 2 by 0.011). The margins are small, and only one fold (fold 4 vs S3) exceeds the ±0.038 rerun range.
   - Recall shifts toward healthy and "both" and away from "LR only", and macro-F1 rises by 0.015–0.018.
   - Validation is unchanged (0.435).
   - Confound: L2 also changes BatchNorm batch composition (8 leaves × 32 patches).

**Consistent across folds?** S1's gain is large but uneven: it is higher in 4/5 folds vs S3 and 3/5 vs A2, it is lower in fold 3 vs both and in fold 4 vs A2, and most of the gain comes from folds 1–2. L2's gain is small but consistent across folds. Nothing else separates from the rerun noise.

## Caveats

- **Single seed per experiment.**
- **Small validation rows:** about 200 leaves each, and the best-over-epochs validation score is optimistic.
- **A1's validation score:** its validation leaves helped fit its scaler.
- **Bootstrap intervals** resample leaves only.
- **No original checkpoints:** the historical original saved none, so it cannot be re-evaluated.

## What to investigate next

1. **Repeat runs:** S1 vs S3 (and L2) with at least 3 seeds each and deterministic cuDNN (`cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`), so reruns measure seed effects only. About 40 min per experiment per seed.
2. **Variety-only baseline:** predict from the variety alone, to measure how much of the 0.40–0.44 pooled balanced accuracy is variety recognition.
3. **Optimise for the within-variety task:** for example per-variety normalisation, or per-variety models. Select on within-variety validation AUC, since that is what pooled metrics hide.
4. **A less noisy selection signal:** nested or larger validation folds.
