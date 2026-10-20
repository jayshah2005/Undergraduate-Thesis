# Spectra exploration: variety vs disease

Do the leaf spectra separate by variety or by disease? Answer: by variety; see `REPORT.md`.

| path | what it is |
|---|---|
| `explore_spectra.py` | the whole analysis (docstring lists every output and method) |
| `REPORT.md` | findings, Baco/Chardonnay result, caveats, next steps |
| `figures/01`–`07` | spectra by variety, by class per variety, difference spectra, PCA (raw, within Baco/Chardonnay, SNV), per-band effect sizes |
| `results/tables.md`, `results/results.json` | all numbers |
| `results/leaf_mean_spectra.csv` | one row per leaf: labels, class, glint share, 204 mean-reflectance columns (the means come from `../shared/spectra.py`) |
| `logs/run.log` | what the last run printed (the tables and the saved paths) |

```bash
cd "claude runs/spectra_exploration"
../../.venv/bin/python -u explore_spectra.py      # ~10 s
```
