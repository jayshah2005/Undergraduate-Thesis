# Disease classification on Chardonnay: report

**Question.** Within one variety (Chardonnay, 250 leaves), can a small CNN that sees the whole leaf
cube detect leafroll or red blotch better than a logistic regression on the leaf's mean spectrum?
And does attention pooling, which can let a few symptomatic patches decide a leaf, beat plain averaging?

**Design.** Each field row tests once, the next row is validation (early stopping), and the other 3 train.
The baseline trains on the same 3 rows, so neither model has more data.

**Short answer.** No reliable gain on either question.
- **Leafroll:** at or near chance (CNN within-row AUC 0.50–0.55, baseline 0.46).
- **Red blotch:** the best CNN reaches 0.64 against the baseline's 0.60, with overlapping intervals.
- **Why the red blotch number looks better than it is:** the margin comes almost entirely from one test row (R3).
- **Attention:** its weights are nearly uniform, so in practice it behaves like mean pooling.
- **Learning:** the network can memorise the training rows, but what it learns does not carry over to a held-out row.
- **Full spectrum vs RGB:** 3 RGB bands do as well as all 204, for the CNN and for the baseline.

Settings: 90° rotations and flips while training, no band or brightness jitter, dropout 0.5, 3 seeds.
Set-up details: `cnn/README.md` and `logistic_regression/README.md`. Full tables: `cnn/results/chardonnay/metrics.md`.

## Checks before training
- **Padding cannot leak into a prediction.** Masked batch norm and masked pooling use leaf positions only.
  Shifting a leaf by 8 px or enlarging the canvas by 32 px changed its logit by at most 5 × 10⁻⁷. The background is
  exactly 0 after both preprocessings.
- **Photos never straddle train and test.** `../leaf_sources.csv` matched every leaf to its camera image by exact
  pixel values: Chardonnay R1 has 12 images of 4 leaves and one of 2; R2–R5 have 5 per image. No row splits an image.
- **No capture-date confound.** All 250 Chardonnay leaves were imaged on the same day (2026-08-26).
- **Every row has positives for both diseases** (red blotch 10–26 per row, leafroll 10–33).

## Results
AUC compares leaves of the same row, so a row's disease rate can't help (0.5 = chance). Each value is the 3-seed
ensemble with a 95% bootstrap interval. Δ is the paired difference to the baseline.

![AUC by model](cnn/figures/chardonnay/auc_comparison.png)

| model | Red blotch AUC | Δ | Leafroll AUC | Δ |
|---|---|---|---|---|
| baseline | 0.602 [0.521, 0.679] | | 0.464 [0.389, 0.540] | |
| band z-score + mean pool | 0.640 [0.564, 0.713] | +0.038 [−0.045, +0.121] | 0.535 [0.458, 0.619] | +0.071 [−0.026, +0.168] |
| band z-score + attention | 0.640 [0.568, 0.709] | +0.038 [−0.045, +0.118] | 0.546 [0.469, 0.624] | +0.082 [−0.008, +0.177] |
| pixel SNV + mean pool | 0.588 [0.513, 0.656] | −0.014 [−0.095, +0.068] | 0.500 [0.422, 0.573] | +0.036 [−0.058, +0.128] |
| pixel SNV + attention | 0.491 [0.412, 0.572] | −0.110 [−0.196, −0.024] | 0.520 [0.446, 0.597] | +0.056 [−0.028, +0.141] |

## What the numbers say
1. **Red blotch: no CNN arm beats the baseline with confidence.** The closest are the band z-score arms, at Δ +0.038.
   On R3 they score 0.86–0.88 against the baseline's 0.64; on the other rows they score 0.52–0.64 against 0.53–0.67.
2. **SNV + attention is below the baseline on red blotch.** It is one of 8 task × arm results, and its seeds range
   from 0.45 to 0.57, so treat this as weak evidence.
3. **Small changes move results as much as the differences between arms.** Single seeds of one arm differ by up to 0.12.
4. **Leafroll stays at or near chance.** No interval on Δ excludes 0. Where the band z-score arms look ahead, it is
   mostly because the baseline is below chance, with 0.20 on R4.
5. **Attention behaves like mean pooling.**
   - **Concentration:** the effective number of positions a leaf's attention covers, 1 / Σ a², is a median 97–99% of
     its valid positions (seed 0).
   - **Heatmaps** (`cnn/figures/chardonnay/attention_maps.png`, fixed scale where 1 = uniform): no
     concentration on a few patches.
   - **Consequence:** the label-noise idea (let symptomatic patches decide) could not be tested. No weighting of
     patches carried over to a new row.

![Attention maps](cnn/figures/chardonnay/attention_maps.png)

## Training diagnostics: underfitting or no transferable signal?
These checks were made on the first runs, which still used jitter. "No better than the baseline" could mean the network
could not learn at all. Two checks separate the two cases:
- **With regularisation** (augmentation, dropout 0.5, weight decay 0.01), training loss barely moves.
  - Leafroll: median 0.75 → 0.70 over the run. ln 2 ≈ 0.69 is a coin flip.
  - Red blotch: 0.93 → 0.84, against about 0.90 for a constant prediction.
- **Capacity test** (one fold, band z-score + attention, no augmentation, no dropout, no weight decay, 150 epochs).
  - The same network memorises the 150 training leaves: training loss 0.84 → 0.001 (leafroll) and 1.08 → 0.09
    (red blotch).
  - But the lowest validation loss comes at epochs 2–16. The more it fits the 3 training rows, the worse it does on
    a new row.

So the network can learn, and is not under-powered. Both are what we would see if, at this sample size, the leaf cubes
carry no disease pattern that holds across rows. A larger dataset, or a much better architecture, could still change
that; this run cannot rule it out.

## Without augmentation
`--no-augmentation` trains without rotations and flips (`cnn/results/chardonnay_no_augmentation/metrics.md`).
It doesn't change the conclusion.
- **Red blotch:** CNN arms 0.571–0.624, against the baseline's 0.602. The best Δ is +0.022 [−0.069, +0.109].
- **Leafroll:** CNN arms 0.496–0.554. Band z-score + attention's Δ, +0.090 [+0.008, +0.182], just clears 0, but its
  own AUC interval [0.476, 0.630] includes chance, and the gap comes from the below-chance baseline.
- **SNV + attention on red blotch:** 0.595 (Δ −0.007), so the significant loss above (−0.110) doesn't hold up
  without augmentation. Treat it as noise.
- **Pooled `any` over all varieties:** best arm 0.549 against 0.523 (`cnn/results/all_no_augmentation/metrics.md`).

## RGB bands only
`--bands rgb` keeps the 3 bands nearest 640, 550 and 460 nm, as the variety models do, for both the CNN and the
baseline (`cnn/results/chardonnay_rgb/metrics.md`, `logistic_regression/results/chardonnay_rgb/metrics.md`).
Everything else is the same as above. SNV then works over 3 values, so the SNV arms and the baseline see little
more than the ratio of red, green and blue.

| model | Red blotch AUC | Δ | Leafroll AUC | Δ |
|---|---|---|---|---|
| baseline (RGB) | 0.626 [0.551, 0.697] | | 0.510 [0.436, 0.587] | |
| band z-score + mean pool | 0.613 [0.537, 0.687] | −0.013 [−0.064, +0.038] | 0.517 [0.443, 0.595] | +0.008 [−0.083, +0.103] |
| band z-score + attention | 0.621 [0.544, 0.694] | −0.004 [−0.060, +0.048] | 0.548 [0.473, 0.623] | +0.038 [−0.043, +0.120] |
| pixel SNV + mean pool | 0.608 [0.530, 0.682] | −0.017 [−0.049, +0.013] | 0.559 [0.487, 0.633] | +0.050 [−0.020, +0.121] |
| pixel SNV + attention | 0.613 [0.535, 0.686] | −0.013 [−0.045, +0.016] | 0.583 [0.513, 0.656] | +0.074 [−0.000, +0.147] |

- **Red blotch:** the RGB models score about the same as the full-spectrum ones. The RGB baseline (0.626) is
  slightly above the 204-band baseline (0.602), and the RGB CNN arms (0.608–0.621) sit just under the best
  full-spectrum arms (0.640). All intervals overlap, so the extra 201 bands add nothing we can detect. R3 still carries
  most of the signal (RGB baseline 0.83 there).
- **Leafroll:** SNV + attention reaches 0.583, Δ +0.074 [−0.000, +0.147]. That is one of 8 results, its interval on Δ
  touches 0, and most of it comes from R2 (0.76 against 0.50–0.57 elsewhere). Treat it as weak evidence. The other
  arms are 0.517–0.559.
- **Pooled `any` over all varieties** (`cnn/results/all_rgb/metrics.md`): the RGB baseline scores 0.554 [0.510, 0.598],
  above both its 204-band version (0.523) and every RGB CNN arm (0.513–0.528, Δ −0.026 to −0.041, intervals include 0).
- **Training ran its course.** Only 1 of the 180 RGB folds reached the 80-epoch cap, unlike the variety RGB CNN.
- **What this means:** whatever disease signal the leaves carry at this sample size is visible in colour. A
  hyperspectral camera adds nothing detectable for these tasks.

## Earlier runs
Their files are deleted, along with the code for the 4-row baseline, the `photo` and `block` folds and brightness
jitter; only these numbers remain. None changed the conclusion.
- **Baseline on 4 rows:** red blotch 0.565 [0.482, 0.646], leafroll 0.460 [0.387, 0.538]. Measured against it, the best
  red blotch Δ was +0.074 [−0.007, +0.162]. The baseline scores higher on red blotch with fewer rows, so its row
  count is itself a source of noise.
- **Random photos in every fold (`photo`):** red blotch, baseline 0.604, CNN arms 0.587–0.620 (best Δ +0.016
  [−0.061, +0.091] against a 4-fold baseline). Leafroll, baseline 0.489, CNN arms 0.46–0.59.
- **Blocks of neighbouring vines in every fold (`block`):** red blotch, baseline 0.577, CNN arms 0.536–0.572 (best Δ −0.005
  [−0.101, +0.095] against a 4-fold baseline). Leafroll, baseline 0.433, CNN arms 0.46–0.53.
- **The best arm changed with the fold design:** SNV + attention was the worst under row folds and the best under `photo`.
- **With jitter (first runs):** red blotch, CNN arms 0.60–0.62 on row folds. Training also scaled each band by 2% and leaf
  brightness by U(0.8, 1.2).
- **Dropout 0.2:** no overall gain across all 24 results of the three designs (mean change −0.004; 9 up, 15 down).

## Context
This matches the archived transfer-learning study (`claude runs/cache/transfer_learning_study`). There,
within-variety AUC stayed at 0.48–0.59 for every approach, including attention MIL over pixel bags and a whole-leaf
ResNet-18. This run differs in three ways:
- one variety, binary tasks;
- native-resolution whole leaves with masked normalisation;
- attention over regions of about 26 px, not single pixels.

None of these changed the picture. The pooled run over all 998 leaves (`any` disease vs healthy) agrees too: best arm
0.545 against the baseline's 0.523, Δ +0.022 [−0.031, +0.074] (`cnn/results/all/metrics.md`).

## For the poster
- **Claim:** "A whole-leaf hyperspectral CNN, trained within Chardonnay, does not reliably detect leafroll or red
  blotch better than the leaf's mean spectrum on field rows it has never seen (best red-blotch CNN 0.64 within rows,
  against 0.60 for the baseline, with overlapping intervals)."
- **Figures:**
  - `cnn/figures/chardonnay/auc_comparison.png` shows the main result.
  - `cnn/figures/chardonnay/attention_maps.png` is a fair illustration only with its caption: the weights
    are nearly uniform.
- **Not supported:** "attention finds symptoms"; "spatial information helps".
- **Caveat to state:** the network can memorise the training leaves, but what it learns does not carry over to
  leaves from other rows. With about 150 training leaves per fold, the result is "no transferable signal at this
  sample size", not proof that no signal exists.

## If you continue
- **Red blotch only.** Leafroll showed no signal. The next step is to strengthen the mean-spectrum signal, not the CNN.
- **Row R3 for red blotch.** Look at those 13 positives and what makes them easy: visible symptoms, or something
  about that row. That is the one place a signal shows up.
- **Label noise.** If many infected leaves have no symptoms, no model can reach high AUC on these labels.
  Visual symptom scores, if any exist, would be a cleaner target.
- **More leaves per variety.** That would do more than any architecture change: the 95% intervals are about ±0.08 wide.

## Reproducing
```bash
cd clean/disease_classification/logistic_regression
../../../.venv/bin/python -u logistic_regression.py              # baseline alone, seconds
cd ../cnn
../../../.venv/bin/python -u run_experiments.py                  # ~5 min on the RTX PRO 5000; runs the baseline too
../../../.venv/bin/python -u report.py
../../../.venv/bin/python -u run_experiments.py --variety all --task any   # pooled run, ~20 min
../../../.venv/bin/python -u report.py --variety all
# RGB bands only: add --bands rgb to each of the commands above (results/<variety>_rgb/)
```
Seeds are fixed, but cuDNN max-pool backward has no deterministic kernel, so reruns can differ slightly
(`torch.use_deterministic_algorithms(..., warn_only=True)`).
