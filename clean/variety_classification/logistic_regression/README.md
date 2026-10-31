# Variety prediction with logistic regression

Each leaf is its mean spectrum → SNV (removes brightness) → per-band standardisation (fit on training leaves) → multinomial logistic regression. C is chosen inside each training split by leave-one-row-out log-loss, over 1e-5 … 1000. Evaluation holds out one field row at a time, so every leaf is predicted by a model that never saw its row.

| setting | choices | meaning |
|---|---|---|
| `--bands` | `all` (default), `rgb` | all 204 bands, or the 3 bands nearest 640 / 550 / 460 nm (actual 640.0 / 548.5 / 461.0 nm) |

```bash
cd clean/variety_classification/logistic_regression
../../../.venv/bin/python -u logistic_regression.py --bands rgb   # one configuration, ~10 s
../../../.venv/bin/python -u logistic_regression.py --compare    # table of every finished configuration
../../../.venv/bin/python -u band_importance.py                  # which wavelengths matter (all bands), ~20 s
```

| output | what it is |
|---|---|
| `results/<bands>/metrics.md` | balanced accuracy with 95% CI, confusion matrix, per-variety recall and ROC AUC, per-row scores |
| `results/<bands>/metrics.json` | the same numbers, read by `--compare` |
| `results/comparison.md` | both configurations side by side: balanced accuracy with 95% CI, the recall of each variety, per-row range |
| `results/band_importance.md`, `figures/band_importance.png` | one-band models, regions, fewest bands, permutation drop |
| `figures/<bands>/` | `confusion_matrix.png`, `coefficients.png` |

Ported from `claude runs/variety_logistic_regression` and `claude runs/variety_logistic_regression_rgb`. The earlier runs also tried raw reflectance, which is not ported. With SNV it gives the same predicted variety for every leaf. Confidence intervals differ by up to 0.003 because the bootstrap now runs in PyTorch. Band-importance permutation drops differ because the shuffle does too, and region scores by up to 0.002 from floating-point rounding in SNV.

Results and what they mean: [`../REPORT.md`](../REPORT.md).
