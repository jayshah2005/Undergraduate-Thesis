# Disease classification: report

**Question.** Can a whole-leaf CNN detect leafroll or red blotch in Chardonnay better than logistic regression on
the leaf's mean spectrum? And do all 204 bands help compared with 3 RGB bands?

**Set-up.** 250 Chardonnay leaves. Each field row is the test set once; both models train on the same 3 other rows.
Score: balanced accuracy at P ≥ 0.5 (0.5 = chance), with a 95% bootstrap interval. CNN = band z-score + attention
pooling, average of 3 seeds.

| Red blotch | Logistic regression | CNN |
|---|---|---|
| Full spectrum (204 bands) | 0.605 [0.547, 0.662] | 0.592 [0.533, 0.648] |
| RGB (3 bands) | 0.619 [0.557, 0.679] | 0.625 [0.567, 0.681] |

| Leafroll | Logistic regression | CNN |
|---|---|---|
| Full spectrum (204 bands) | 0.537 [0.477, 0.596] | 0.595 [0.544, 0.647] |
| RGB (3 bands) | 0.588 [0.536, 0.644] | 0.572 [0.524, 0.620] |

**What it shows**
- Red blotch is detected a little better than chance (about 0.60); leafroll barely (0.54–0.60).
- The CNN is not reliably better than logistic regression: every CNN − baseline interval includes 0.
- RGB does as well as the full spectrum, so the extra bands add nothing we can detect at this sample size.
- The other three CNN versions (`cnn/results/chardonnay/metrics.md`) give the same picture.
- All 998 leaves, any disease vs healthy (`cnn/results/all/metrics.md`): CNN up to 0.602 vs 0.556, but within each
  variety the scores are near chance (0.39–0.66), so the gain mostly comes from telling varieties apart.

Figures: `cnn/figures/chardonnay/balanced_accuracy.png` (and `chardonnay_rgb/`). How to rerun: `cnn/README.md`.
