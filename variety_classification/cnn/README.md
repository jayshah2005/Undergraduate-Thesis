# Variety prediction with a whole-leaf CNN

Does a small CNN that sees the whole leaf do better than logistic regression on its mean spectrum, and do 204 bands beat RGB?

| path | what it is |
|---|---|
| `model.py` | `VarietyCNN`: 1×1 spectral conv → four 3×3 conv layers with masked batch norm → masked mean pooling → 4 logits |
| `run_experiments.py` | padding check → train every (seed, fold) of one configuration; skips fold files already on disk |
| `report.py` | metrics, paired differences and figures for both inputs, with logistic regression as the comparison |
| `plot_architecture.py` | poster schematic of the network → `figures/architecture.png` |

| setting | choices | meaning |
|---|---|---|
| `--bands` | `all` (default), `rgb` | all 204 bands, or the 3 bands nearest 640 / 550 / 460 nm. Only the input of the first layer changes |

## Set-up

- **Input:** each pixel gets SNV over its own bands, which removes brightness. Each leaf is centred on one 212 × 212 canvas (the largest leaf side, rounded up to a multiple of 4). Padding is 0 and a mask marks leaf pixels. Normalisation and pooling use leaf positions only, so padding can't change a prediction; each run checks this at start-up.
- **Splits:** fold k tests on row Rk of every variety (each variety has its own rows R1–R5), the next row is validation (early stopping), and the other 3 train. 5 folds × 3 seeds = 15 trainings per configuration.
- **Training:** batch 8, AdamW (lr 1e-3, weight decay 1e-2), dropout 0.5, cross-entropy, up to 80 epochs with patience 15. Augmentation is 90° rotations and flips, with no brightness or band jitter. `--no-augmentation` (on both `run_experiments.py` and `report.py`) trains without it, into `results/<bands>_no_augmentation/`, and the report files get the same suffix.
- **Comparison:** logistic regression with the same split: it fits on the same 3 rows and chooses C on the validation row, so neither model has more data.

```bash
cd variety_classification/cnn
../../.venv/bin/python -u run_experiments.py --bands rgb --smoke   # ~40 s: 1 fold, 3 epochs, results/smoke/
../../.venv/bin/python -u run_experiments.py --bands all           # resumable
../../.venv/bin/python -u run_experiments.py --bands rgb
../../.venv/bin/python -u report.py                                # results/<bands>/{final_report,metrics}.md + figures/; needs both runs
../../.venv/bin/python -u plot_architecture.py                     # figures/architecture.png
```

| output | what it is |
|---|---|
| `results/<bands>/seed<s>/fold<k>.npz` | test probabilities and loss curves of one training |
| `results/<bands>/config.json` | settings; a folder only resumes runs made with the same ones |
| `results/<bands>/final_report.md` | balanced accuracy of both models with 95% CI, Δ vs logistic regression, per-fold scores; recall per variety, single seeds, 204 bands − RGB |
| `results/<bands>/metrics.md` | every metric of both models (accuracy, macro F1, per-variety sensitivity, specificity, precision, F1, confusion counts) |
| `figures/scores.png`, `figures/confusion_<bands>.png` | all models compared; the CNN's confusion matrices |
| `logs/<bands>.log` | progress of the last training: one line per seed |

Results and what they mean: [`../REPORT.md`](../REPORT.md).
