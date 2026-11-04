# Logistic regression on the mean spectrum

The disease baseline: logistic regression on each leaf's average spectrum (SNV, then standardised). The CNNs in
`../within_variety_disease_cnn/` and `../cnn/` have to beat it.

| file | what it is |
|---|---|
| `logistic_regression.py` | leaves, labels and folds (shared with the CNN), the model, `metrics.md` |
| `scoring.py` | balanced accuracy and its bootstrap interval |
| `results/<variety>[_rgb]/metrics.md` | shared format: the macro average (mean of leafroll and red-blotch balanced accuracy), then each task, with 95% CI, per-fold scores and (for a pool) per variety |

**Folds.** Fold k tests on row Rk; in a pool that is Rk of every variety, since each variety has its own rows R1–R5. The model trains on 3 other rows; the remaining row is the CNN's
validation row, unused here, so both models train on the same leaves.

**Score.** Balanced accuracy over every fold's test leaves: a leaf counts as positive at P ≥ 0.5, and the score is the mean of
the accuracy on positive and on negative leaves (0.5 = chance). The 95% interval resamples leaves 2000 times.
The macro average is the mean of the leafroll and red-blotch balanced accuracies, so each disease counts equally;
its interval resamples leaves within each variety × row × disease-combination cell, so both diseases see the same leaves.

**Tasks.** `leafroll`, `red_blotch`, `both` (both diseases vs the rest) and `any` (any disease vs healthy). A variety
with fewer than 5 positive or 5 negative leaves sits out that task (e.g. Cab Franc for leafroll).

**Varieties.** `--variety` takes one variety (default Chardonnay) or a pool: `all` or `baco_chardonnay_vidal`.

**Bands.** `--bands all` (default) uses all 204 bands; `--bands rgb` only the bands nearest 640, 550 and 460 nm.

```bash
cd clean/disease_classification/logistic_regression
../../../.venv/bin/python -u logistic_regression.py                          # Chardonnay → results/chardonnay/
../../../.venv/bin/python -u logistic_regression.py --variety all --bands rgb # all leaves, RGB → results/all_rgb/
```
