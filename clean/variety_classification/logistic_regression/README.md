# Variety prediction with logistic regression

Each leaf is its mean spectrum → SNV (removes brightness) → per-band standardisation (fit on training leaves) → multinomial logistic regression. C is chosen inside each training split by leave-one-row-out log-loss, over 1e-5 … 1000. Evaluation holds out one field row at a time, so every leaf is predicted by a model that never saw its row.

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
