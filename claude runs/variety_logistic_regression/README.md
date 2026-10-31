# Variety prediction with logistic regression

Predicts each leaf's variety (Baco / Cab Franc / Chardonnay / Vidal) from its mean spectrum. The full pipeline is in the script's docstring:

1. SNV, which removes brightness.
2. Per-band standardisation.
3. Multinomial logistic regression, with C chosen inside each training split.

It is evaluated on the repo's 5 row folds: each row is held out once.

```bash
cd "claude runs/variety_logistic_regression"
../../.venv/bin/python -u variety_logistic_regression.py                 # SNV spectra (default), ~10 s
../../.venv/bin/python -u variety_logistic_regression.py --spectra raw   # raw spectra, for comparison
```

| output | what it is |
|---|---|
| `results/variety_metrics.md` / `.json` | confusion matrix, balanced accuracy with 95% CI, macro-F1, per-variety precision / recall / F1 / ROC AUC, per-fold results |
| `results/variety_oof.csv` | per leaf: true variety, the 4 probabilities, predicted variety |
| `figures/confusion_matrix.png` | the confusion matrix |
| `figures/coefficients.png` | the weight the model puts on each band, per variety |
| `logs/run.log` | what the last run printed (the metrics markdown) |
| files ending in `_raw` | the same outputs (and `logs/run_raw.log`) for `--spectra raw` |

## Result

| spectra | balanced accuracy (95% CI) | macro-F1 | per-fold range |
|---|---|---|---|
| SNV | **0.939** (0.924–0.954) | 0.940 | 0.88–0.99 |
| raw | 0.920 (0.904–0.936) | 0.920 | 0.82–0.99 |

Chance is 0.25.

- **Where the errors are:** almost all involve Vidal. 21 Baco and 15 Cab Franc leaves were called Vidal, and 13 Vidal leaves were called Cab Franc.
- **What the model uses:** after removing brightness it still identifies the variety, so it relies on spectral shape. No single band does the job (best one-band model: 691 nm, balanced accuracy 0.71). Green (500–600 nm) or red (600–680 nm) alone already reach ~0.89; the full 204-band score is 0.94. Details: `results/band_importance.md` and `figures/band_importance.png`.
- **Caveat:** "variety" also carries anything acquired together with it (imaging day, lighting). A high score shows the spectra separate the varieties, not that cultivar biology alone is the cause.
