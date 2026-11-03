# Variety prediction with logistic regression

Each leaf is its mean spectrum → SNV (removes brightness) → per-band standardisation (fit on training leaves) → multinomial logistic regression. It uses the same split as the CNN, so neither model has more data: one field row is the test set, the next is validation, and the other 3 train. Each C over 1e-5 … 1000 is fit on the 3 training rows, and the one with the lowest log-loss on the validation row is kept. Every leaf is predicted by a model that never saw its row.

| file | what it is |
|---|---|
| `logistic_regression.py` | features, the row split both models use, fitting, tables and the confusion figure |
| `scoring.py` | balanced accuracy and its bootstrap interval, recall, ROC AUC, confusion matrix; used here and in `../cnn/report.py` |

| setting | choices | meaning |
|---|---|---|
| `--bands` | `all` (default), `rgb` | all 204 bands, or the 3 bands nearest 640 / 550 / 460 nm (actual 640.0 / 548.5 / 461.0 nm) |

```bash
cd clean/variety_classification/logistic_regression
../../../.venv/bin/python -u logistic_regression.py --bands rgb   # one configuration, ~10 s
../../../.venv/bin/python -u logistic_regression.py --compare    # table of every finished configuration
```

| output | what it is |
|---|---|
| `results/<bands>/metrics.md` | balanced accuracy with 95% CI, confusion matrix, per-variety recall and ROC AUC, per-row scores |
| `results/<bands>/metrics.json` | the same numbers, read by `--compare` |
| `results/comparison.md` | both configurations side by side: balanced accuracy with 95% CI, the recall of each variety, per-row range |
| `figures/<bands>/` | `confusion_matrix.png` |

Ported from `claude runs/variety_logistic_regression` and `claude runs/variety_logistic_regression_rgb`. The earlier runs also tried raw reflectance, which is not ported. With SNV it gives the same predicted variety for every leaf. Confidence intervals differ by up to 0.003 because the bootstrap now runs in PyTorch.

Results and what they mean: [`../REPORT.md`](../REPORT.md).
