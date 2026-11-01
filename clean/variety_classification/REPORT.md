# Variety classification: report

**Question.** How well can a leaf's variety be predicted from held-out field rows, does a whole-leaf CNN beat logistic regression on the mean spectrum, and do 204 bands beat RGB? Every model here uses SNV, which removes brightness.

**Short answer.**
- **Most dependable model:** logistic regression on all 204 bands, trained on 4 rows: balanced accuracy 0.939 (chance 0.25). It has no epoch count to choose and gives the same answer every time.
- **CNN vs logistic regression:** logistic regression is better. With the epoch count chosen inside each training split, a single CNN on 204 bands averages 0.814 against 0.939. Only a 3-seed ensemble comes level, at 0.925 (Δ −0.014 [−0.032, +0.004]).
- **204 bands vs RGB:** 204 bands win by a wide margin in logistic regression. The CNN gives no firm answer. Single RGB models average higher (0.872 vs 0.814), but only on 2 of 3 seeds and with no interval. Ensembles favour 204 bands (0.925 vs 0.876, Δ +0.049 [+0.027, +0.072]).

## Results

Balanced accuracy, with 95% intervals from resampling leaves. CNN numbers are 3-seed ensembles.

| model | 204 bands | RGB | Δ 204 bands − RGB [95% CI] |
|---|---|---|---|
| CNN, 4 rows, epochs chosen inside each split | 0.925 (0.909–0.940) | 0.876 (0.857–0.896) | +0.049 [+0.027, +0.072] |
| CNN, 4 rows, no validation, 37 epochs | 0.949 (0.936–0.961) | 0.838 (0.816–0.860) | +0.111 [+0.088, +0.134] |
| CNN, 3 rows, early stopping on a fourth | 0.898 (0.880–0.916) | 0.824 (0.799–0.846) | +0.074 [+0.049, +0.100] |
| Logistic regression, same 3 rows | 0.849 (0.827–0.871) | 0.488 (0.457–0.517) | +0.361 [+0.328, +0.394] |
| Logistic regression, 4 rows | 0.939 (0.924–0.952) | 0.500 (0.468–0.530) | +0.439 [+0.408, +0.470] |

![Scores](cnn/figures/scores.png)

CNN points in the figure are ensembles, so they do not show the single-model results below.

By held-out row:

| model | R1 | R2 | R3 | R4 | R5 | mean without R1 |
|---|---|---|---|---|---|---|
| CNN, 204 bands, 4 rows, epochs chosen inside each split | 0.845 | 0.950 | 0.975 | 0.940 | 0.915 | 0.945 |
| CNN, RGB, 4 rows, epochs chosen inside each split | 0.885 | 0.865 | 0.915 | 0.870 | 0.845 | 0.874 |
| CNN, 204 bands, 4 rows, 37 epochs | 0.915 | 0.970 | 0.964 | 0.960 | 0.935 | 0.957 |
| CNN, 204 bands, 3 rows, early stopping | 0.595 | 0.975 | 0.975 | 0.975 | 0.970 | 0.974 |
| CNN, RGB, 4 rows, 37 epochs | 0.880 | 0.875 | 0.855 | 0.825 | 0.755 | 0.828 |
| CNN, RGB, 3 rows, early stopping | 0.730 | 0.870 | 0.905 | 0.820 | 0.795 | 0.848 |
| Logistic regression, 204 bands, same 3 rows | 0.610 | 0.890 | 0.970 | 0.935 | 0.840 | 0.909 |
| Logistic regression, 204 bands, 4 rows | 0.905 | 0.960 | 0.990 | 0.960 | 0.880 | 0.948 |

## Epochs chosen inside each training split

For each test row, 4 trials trained on 3 of the other rows and validated on the fourth, for 80 epochs each. The epoch E with the lowest average validation loss was then used, unchanged, to train 3 final models on all 4 rows. The rules were fixed before any test score existed, and the headline is the average of the 5 row scores for a single model.

| input | single model: 5-row average, by seed | single model: mean | 3-seed ensemble | E for R1, R2, R3, R4, R5 |
|---|---|---|---|---|
| 204 bands | 0.866, 0.806, 0.771 | **0.814** | 0.925 | 67, 34, 76, 64, 69 |
| RGB | 0.853, 0.884, 0.880 | **0.872** | 0.876 | 64, 70, 77, 80, 57 |
| Logistic regression, 204 bands, for reference | | 0.939 | | none needed |

- **Single 204-band models are unreliable.** Their row scores run from 0.599 to 1.000, against 0.795 to 0.945 for RGB. A poor model can lose most of one variety: seed 2 on R3 finds 8% of Chardonnay leaves, and seed 0 on R4 finds 6%. Averaging 3 seeds repairs most of this, which is why 204 bands lead RGB only as an ensemble. Paired by seed, RGB is higher on 2 of 3 (0.866 vs 0.853, 0.806 vs 0.884, 0.771 vs 0.880).
- **E probably marks a dip in a noisy curve.** The 204-band average validation loss reaches 0.10–0.60 at E but averages 0.79–1.80 over the last 10 epochs, and it moves by a median 0.54–0.87 per epoch. RGB is far steadier: 0.27–0.45 at E, 0.32–0.54 over the last 10, and 0.04–0.06 per epoch. A fresh model stopped at E need not be in the same dip. Training loss is low throughout (0.05–0.17 at the last epoch), so this isn't underfitting. The cause of the noise wasn't investigated.
- **One trial can move E a long way.** The 204-band R2 split chose 34, but would have chosen 59 without its trial validated on R1. That trial trains on neither R1 nor R2 (see point 3 below), and its loss of 2.05 at epoch 34 is far above the other three (0.02–0.25). The rule was left as fixed.
- **RGB may want more epochs.** R4 chose 80, the cap. Moving from 37 epochs to these counts raised RGB's single-model mean from 0.838 to 0.872.
- **204 bands did not benefit from the longer runs.** Compared with 37 epochs, the ensemble fell from 0.949 to 0.925 and the single-model mean from 0.834 to 0.814, a change within seed noise.
- **The headline has no interval.** With 3 seeds per input, its spread is shown by seed rather than as a 95% interval.

## What the numbers say

1. **The no-validation CNN depends on the number of epochs.** The counts were first fixed in advance at 8 (204 bands) and 37 (RGB), each 3/4 of the earlier median best epoch. On that pair RGB beats 204 bands (Δ −0.153 [−0.188, −0.118]), but only because 8 epochs is too few: training loss is still 0.32 there and reaches 0.12 by epoch 37.
   - The sensitivity runs flip the answer. At 37 epochs each, 204 bands win by +0.111. At 8 epochs each, RGB wins (0.685 vs 0.791, Δ −0.105 [−0.141, −0.070]). The 8-epoch run files have since been deleted; these numbers remain here.
   - The likely reason for the failed choice: early stopping keeps the best checkpoint from noisy epochs, so its median is not a good place to stop and keep the last weights.
   - The 37-epoch 204-band run wasn't chosen in advance, and the choice between 8 and 37 here was seen on test rows, so 0.949 is optimistic. The run with epochs chosen inside each split replaces it as the clean number.
2. **The ensemble hides large seed noise.** At 37 epochs and 204 bands, single seeds score 0.826, 0.932 and 0.744, while their ensemble scores 0.949. With epochs chosen inside each split they score 0.866, 0.806 and 0.771, and their ensemble 0.925. The early-stopping seeds were 0.857, 0.856 and 0.914. One trained CNN is much less dependable than the ensemble number suggests.
3. **Rows R1 and R2 need each other in training.** Training on all 4 rows supports this: the 204-band CNN on R1 goes from 0.595 to 0.915. The CNN's validation row costs it training data, and with early stopping, test row R1 is the only fold that trains on neither R1 nor R2.
   - Logistic regression trained on R3, R4 and R5 scores 0.61 on R1 and 0.655 on R2.
   - Adding the other of the two raises that to 0.905 on R1 and 0.96 on R2.
   - This explains both the CNN's 0.595 on R1 and most of logistic regression's fall from 0.939 to 0.849. Why these two rows are alike wasn't investigated.
4. **The RGB gap is partly caused by SNV.** With 3 values, SNV leaves very little: two of the three degrees of freedom go to its centring and scaling. The earlier raw-reflectance run in `claude runs/variety_logistic_regression_rgb` reached 0.859 with RGB, so 0.500 isn't what RGB can do in general.
5. **The CNN on RGB probably isn't using colour alone.** Its ensembles reach 0.82–0.88 while logistic regression on the same 3 bands reaches 0.49–0.50, so the extra must come from spatial information: pattern within the leaf, or leaf size and shape. Leaf size alone (pixel count, height, width) scores 0.654 in logistic regression. No test here separates these.
6. **Intervals are narrower than the real uncertainty.** They resample leaves, but leaves in a row aren't independent, there are only 5 rows, and they ignore seed and epoch choice. Points 1–3 show how much those matter.
7. **Adding the mean spectrum to the CNN's last layer did not help.** Trained like the 3-row early-stopping run, with no weight decay on the spectrum weights, the ensemble's change was +0.000 [−0.016, +0.016] on 204 bands and +0.009 [−0.005, +0.025] on RGB, which missed the rule fixed in advance. The code and run files have since been deleted; these numbers remain here.

## Caveat: variety is confounded with imaging day

Baco was imaged on 2026-08-12, Chardonnay on 2026-08-26, and Cab Franc and Vidal on 2026-09-03. SNV removes brightness, but not any spectral difference between sessions, such as lighting colour. These scores show that the varieties can be told apart in this dataset, not that the cause is cultivar biology alone. Rows are held out, but no imaging day is, since each variety has only one.

## For the poster

- **Claim:** "On held-out rows, a leaf's SNV mean spectrum predicts its variety with balanced accuracy 0.94 (chance 0.25). RGB alone reaches 0.50 under the same model."
- **State alongside it:** the imaging-day caveat, and that the RGB figure is for SNV-normalised RGB.
- **Not supported:** "the CNN beats logistic regression", "skipping validation improves the CNN", "adding the mean spectrum improves the CNN", "204 bands beat RGB in the CNN" or "RGB beats 204 bands in the CNN" (ensembles and single models disagree), and "hyperspectral beats RGB by 0.44 in general".
