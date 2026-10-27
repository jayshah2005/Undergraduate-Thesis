# Variety prediction with logistic regression (RGB only)

Same nested row-fold logistic regression as `../variety_logistic_regression/`, but each leaf is three numbers: the mean reflectance at the Specim IQ bands nearest **640 / 550 / 460 nm** (actual centres 639.4 / 550.7 / 459.1 nm). Those are the same pseudo-RGB picks as the archived ResNet adaptor.

The point is a control: how much of the 204-band variety score is already in a colour photograph?

```bash
cd "claude runs/variety_logistic_regression_rgb"
../../.venv/bin/python -u variety_logistic_regression_rgb.py                 # SNV on the RGB triplet (default)
../../.venv/bin/python -u variety_logistic_regression_rgb.py --spectra raw   # raw R, G, B
```

| output | what it is |
|---|---|
| `results/variety_metrics.md` / `.json` | confusion matrix, balanced accuracy with 95% CI, macro-F1, per-variety precision / recall / F1 / ROC AUC, per-fold results, band indices |
| `results/variety_oof.csv` | per leaf: true variety, the 4 probabilities, predicted variety |
| `figures/confusion_matrix.png` | the confusion matrix |
| `figures/coefficients.png` | weight on R, G, B per variety (bar chart) |
| `logs/run.log` | what the last run printed |
| files ending in `_raw` | the same outputs (and `logs/run_raw.log`) for `--spectra raw` |

## Result

Chance is 0.25. The 204-band numbers are from `../variety_logistic_regression/`.

| features | spectra | balanced accuracy (95% CI) | macro-F1 | per-fold range |
|---|---|---|---|---|
| 204 bands | SNV | **0.939** (0.924–0.954) | 0.940 | 0.88–0.99 |
| 204 bands | raw | 0.920 (0.904–0.936) | 0.920 | 0.82–0.99 |
| RGB (3 bands) | raw | **0.848** (0.825–0.870) | 0.848 | 0.81–0.90 |
| RGB (3 bands) | SNV | 0.506 (0.477–0.536) | 0.504 | 0.42–0.60 |

- **Raw RGB is already enough to tell varieties apart well** (0.85 vs 0.92 with all 204 bands). Most of the raw-spectrum variety signal is visible colour / brightness, not the NIR.
- **SNV kills the RGB model** (0.51). With only three channels, SNV throws away brightness and leaves almost no shape. The 204-band SNV model stays at 0.94 because it still has the green peak, red edge, and the rest of the curve.
- **Where raw RGB errors are:** Cab Franc vs Vidal (48 Cab Franc → Vidal, 23 Vidal → Cab Franc), plus some Baco/Chardonnay mix-ups.
- **Caveat:** same as the 204-band run — variety is confounded with imaging day and lighting. Raw RGB in particular can use brightness from those sources.
