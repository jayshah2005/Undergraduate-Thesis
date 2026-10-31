# Variety prediction with a whole-leaf CNN

Does a small CNN that sees the whole leaf do better than logistic regression on its mean spectrum, and do 204 bands beat RGB?

| path | what it is |
|---|---|
| `model.py` | `VarietyCNN`: 1×1 spectral conv → four 3×3 conv layers with masked batch norm → masked mean pooling → 4 logits |
| `run_experiments.py` | padding check → train every (seed, test row) of one configuration; skips fold files already on disk |
| `run_nested.py` | chooses the epoch count inside each training split, then trains and scores the final models |
| `report.py` | metrics, paired differences and figures for both inputs, with logistic regression as the comparison |

| setting | choices | meaning |
|---|---|---|
| `--bands` | `all` (default), `rgb` | all 204 bands, or the 3 bands nearest 640 / 550 / 460 nm. Only the input of the first layer changes |
| `--no-validation --epochs N` | off by default | train on all 4 non-test rows for exactly N epochs, keeping the last weights, with no early stopping. Results go to `results/<bands>_no_val_e<N>/` |

## Choosing the epoch count inside each training split

Fixed before any test score was seen. For each test row:

1. Set the test row aside; no trial touches it.
2. Run 4 trials on the other rows. Each trains a fresh model on 3 of them and validates on the 4th for 80 epochs (1 seed).
3. Average the 4 validation-loss curves. The epoch with the lowest average is E, used as it is, with no rescaling.
4. Train fresh final models on all 4 rows for exactly E epochs, one per seed (3 seeds), with no validation.
5. Score them once on the test row.

Headline: the 5 row scores averaged, for each single model, then the mean of those. The 3-seed ensemble is reported too, to compare with the other runs. Results go to `results/<bands>_nested/`, with E per row in `chosen_epochs.json`.

## Set-up

- **Input:** each pixel gets SNV over its own bands, which removes brightness. Each leaf is centred on one 212 × 212 canvas (the largest leaf side, rounded up to a multiple of 4). Padding is 0 and a mask marks leaf pixels. Normalisation and pooling use leaf positions only, so padding can't change a prediction; each run checks this at start-up.
- **Splits:** one field row is the test set, the next is validation (early stopping), and the other 3 train. 5 test rows × 3 seeds = 15 trainings per configuration.
- **Training:** batch 8, AdamW (lr 1e-3, weight decay 1e-2), dropout 0.5, cross-entropy, up to 80 epochs with patience 15. Augmentation is 90° rotations and flips, with no brightness or band jitter.
- **Without validation:** all 4 non-test rows train, for a number of epochs fixed in advance. The score depends heavily on that number (`../REPORT.md`).
- **Comparison:** logistic regression trained on the same 3 rows, so neither model has more data, and on the usual 4.

```bash
cd clean/variety_classification/cnn
../../../.venv/bin/python -u run_experiments.py --bands rgb --smoke   # ~40 s: 1 fold, 3 epochs, results/smoke/
../../../.venv/bin/python -u run_experiments.py --bands all           # resumable
../../../.venv/bin/python -u run_experiments.py --bands rgb
../../../.venv/bin/python -u run_experiments.py --bands all --no-validation --epochs 37
../../../.venv/bin/python -u run_nested.py --bands all                # ~1 h each; --smoke for a quick check
../../../.venv/bin/python -u run_nested.py --bands rgb
../../../.venv/bin/python -u report.py                                # results/metrics.md + figures/; needs every run it compares
```

| output | what it is |
|---|---|
| `results/<bands>/seed<s>/fold<k>.npz` | test probabilities and loss curves of one training |
| `results/<bands>/config.json` | settings; a folder only resumes runs made with the same ones |
| `results/metrics.md` | every model, paired differences, per-seed and per-row scores |
| `figures/scores.png`, `figures/confusion_<bands>_no_val_e37.png` | all models compared; confusion matrices at 37 epochs |
| `logs/<run>.log` | progress of the last training: one line per seed, or per test row for `run_nested.py` |

Results and what they mean: [`../REPORT.md`](../REPORT.md).
