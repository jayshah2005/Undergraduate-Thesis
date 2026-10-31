# Whole-leaf CNN with attention pooling on Chardonnay: report

**Question.** Within one variety (Chardonnay, 250 leaves), can a small CNN that sees the whole leaf
cube detect leafroll or red blotch better than a logistic regression on the leaf's mean spectrum?
And does attention pooling, which can let a few symptomatic patches decide a leaf, beat plain averaging?

**Short answer.** No reliable gain on either question.
- **Leafroll:** at or near chance under every fold design (CNN within-row AUC 0.46–0.59).
- **Red blotch:** the best CNN reaches 0.57–0.64, level with the baseline's 0.57–0.60, with overlapping intervals.
- **Why the red blotch number looks better than it is:** the `row` margin comes almost entirely from one test row (R3).
- **Attention:** its weights are nearly uniform, so in practice it behaves like mean pooling.
- **Learning:** the network can memorise the training rows, but what it learns does not carry over to a held-out row.

Settings: no band or brightness jitter, dropout 0.5, 3 seeds, folds built from the real camera images.
Set-up details: `README.md`. Full tables: `results/<scheme>_no_brightness_jitter/metrics.md` and
`results/fold_comparison_no_brightness_jitter.md`.

## Checks before training
- **Padding cannot leak into a prediction.** Masked batch norm and masked pooling use leaf positions only.
  Shifting a leaf by 8 px or enlarging the canvas by 32 px changed its logit by at most 5 × 10⁻⁷. The background is
  exactly 0 after both preprocessings.
- **Photos never straddle train and test.** `../../leaf_sources.csv` matched every leaf to its camera image by exact
  pixel values: Chardonnay R1 has 12 images of 4 leaves and one of 2; R2–R5 have 5 per image. No fold splits an image.
- **No capture-date confound.** All 250 Chardonnay leaves were imaged on the same day (2026-08-26).
- **Every row has positives for both diseases** (red blotch 10–26 per row, leafroll 10–33).

## Results
AUC compares leaves of the same fold **and** the same row, so a row's disease rate can't help (0.5 = chance). Each value
is the 3-seed ensemble with a 95% bootstrap interval. The baseline trains on all 4 non-test folds.

- **`row`:** the test row is never seen in training.
- **`photo`:** each fold holds random photos from every row.
- **`block`:** each fold holds about 10 neighbouring vines of every row, so held-out vines sit next to each other rather
  than between training vines. This weakens two possible shortcuts: virus spreading from vine to vine, and drift in
  the order photos were taken.

![Fold designs compared](figures/fold_comparison_no_brightness_jitter.png)

| Within-row AUC [95% CI] | `row` | `photo` | `block` |
|---|---|---|---|
| **Red blotch**, baseline | 0.565 [0.482, 0.646] | 0.604 [0.533, 0.671] | 0.577 [0.498, 0.660] |
| band z-score + mean pool | 0.640 [0.564, 0.713] | 0.587 [0.514, 0.654] | 0.572 [0.493, 0.655] |
| band z-score + attention | 0.640 [0.568, 0.709] | 0.614 [0.541, 0.687] | 0.559 [0.477, 0.644] |
| pixel SNV + mean pool | 0.588 [0.513, 0.656] | 0.591 [0.526, 0.659] | 0.559 [0.475, 0.640] |
| pixel SNV + attention | 0.491 [0.412, 0.572] | 0.620 [0.553, 0.687] | 0.536 [0.457, 0.615] |
| **Leafroll**, baseline | 0.460 [0.387, 0.538] | 0.489 [0.408, 0.572] | 0.433 [0.355, 0.509] |
| CNN arms (range) | 0.50–0.55 | 0.46–0.59 | 0.46–0.53 |

## What the numbers say
1. **Red blotch: no CNN arm beats the baseline with confidence.**
   - `row`: closest, at Δ +0.074 [−0.007, +0.162]. On R3 the band z-score arms score 0.86–0.88; on the other rows 0.52–0.64.
   - `photo`: level (best Δ +0.016 [−0.061, +0.091]).
   - `block`: level or below (best Δ −0.005 [−0.101, +0.095]).
2. **The best arm changes with the design.** Band z-score leads under `row`, SNV + attention under `photo`, and
   band z-score + mean pool under `block`. SNV + attention is both the worst (`row`) and the best (`photo`).
3. **Small changes move results as much as the differences between arms.** Moving 4 leaves between folds took band
   z-score + attention under `block` from 0.603 to 0.559. Single seeds of one arm differ by up to 0.07.
4. **Leafroll stays at or near chance.** SNV + mean pool under `photo` reaches 0.594 [0.522, 0.670], Δ +0.105
   [+0.004, +0.203]. It isn't a finding: the same arm scored 0.492 with an earlier random photo draw and 0.459
   under `block`, and it is one of 24 task × arm × design results. Where the band z-score arms beat the leafroll baseline,
   it is mostly because the baseline is below chance (0.43–0.46).
5. **Attention behaves like mean pooling.**
   - **Concentration:** the effective number of positions a leaf's attention covers, 1 / Σ a², is a median 97–99% of
     its valid positions (seed 0).
   - **Heatmaps** (`figures/row_no_brightness_jitter/attention_maps.png`, fixed scale where 1 = uniform): no
     concentration on a few patches.
   - **Consequence:** the label-noise idea (let symptomatic patches decide) could not be tested. No weighting of
     patches carried over to a new row.

![Attention maps](figures/row_no_brightness_jitter/attention_maps.png)

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

## Earlier runs
Their files are deleted; only these numbers remain. None changed the conclusion.
- **With jitter (first runs):** red blotch, CNN arms 0.60–0.62 under `row`. Training also scaled each band by 2% and leaf
  brightness by U(0.8, 1.2).
- **Dropout 0.2:** no overall gain across all 24 results (mean change −0.004; 9 up, 15 down). Run on the earlier folds.
- **Earlier folds:** they assumed 5 leaves per photo, which split 2 R1 photos under `block` and 4 under `photo`.
  Correcting them moved 4 leaves under `block` and redrew `photo`.

## Context
This matches the archived transfer-learning study (`claude runs/cache/transfer_learning_study`). There,
within-variety AUC stayed at 0.48–0.59 for every approach, including attention MIL over pixel bags and a whole-leaf
ResNet-18. This run differs in three ways:
- one variety, binary tasks;
- native-resolution whole leaves with masked normalisation;
- attention over regions of about 26 px, not single pixels.

None of these changed the picture.

## For the poster
- **Claim:** "A whole-leaf hyperspectral CNN, trained within Chardonnay, does not reliably detect leafroll or red
  blotch better than the leaf's mean spectrum. This holds whether the test vines come from unseen rows, random photos,
  or blocks of neighbouring vines (best red-blotch CNN 0.57–0.64 within rows, against 0.57–0.60 for the baseline, with
  overlapping intervals)."
- **Figures:**
  - `figures/fold_comparison_no_brightness_jitter.png` shows the main result, under all three fold designs at once.
  - `figures/row_no_brightness_jitter/auc_comparison.png` shows the row design alone.
  - `figures/row_no_brightness_jitter/attention_maps.png` is a fair illustration only with its caption: the weights
    are nearly uniform.
- **Not supported:** "attention finds symptoms"; "spatial information helps".
- **Caveat to state:** the network can memorise the training leaves, but what it learns does not carry over to
  leaves from other rows. With about 150 training leaves per fold, the result is "no transferable signal at this
  sample size", not proof that no signal exists.

## If you continue
- **Red blotch only.** Leafroll showed no signal under any design. The next step is to strengthen the mean-spectrum
  signal, not the CNN.
- **Row R3 for red blotch.** Look at those 13 positives and what makes them easy: visible symptoms, or something
  about that row. That is the one place a signal shows up.
- **Label noise.** If many infected leaves have no symptoms, no model can reach high AUC on these labels.
  Visual symptom scores, if any exist, would be a cleaner target.
- **More leaves per variety.** That would do more than any architecture change: the 95% intervals are about ±0.08 wide.

## Reproducing
```bash
cd clean/disease_classification/attention_mil
for f in row photo block; do
  ../../../.venv/bin/python -u run_experiments.py --folds $f --no-brightness-jitter   # ~6 min each on the RTX PRO 5000
  ../../../.venv/bin/python -u report.py --folds $f --no-brightness-jitter
done
../../../.venv/bin/python -u report.py --compare --no-brightness-jitter
```
Seeds are fixed, but cuDNN max-pool backward has no deterministic kernel, so reruns can differ slightly
(`torch.use_deterministic_algorithms(..., warn_only=True)`).
