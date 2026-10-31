# Disease differences inside each variety

Are healthy, red-blotch-only (RB) and leafroll-only (LR) leaves spectrally different within Baco, Cab Franc, Chardonnay and Vidal? Leaves with both viruses are left out. Each leaf is represented by its mean spectrum (204 bands). Every test is run on the raw spectra and after SNV, which removes brightness and keeps the shape.

```bash
cd "claude runs/disease_differences"
../../.venv/bin/python -u disease_differences.py      # ~10 s
```

| output | what it is |
|---|---|
| `figures/<variety>.png` | (a) raw group means ± 1 SD, (b) the same after SNV, (c) per-band Cohen's d for each pair (SNV), (d) PCA of the variety's leaves |
| `figures/summary_effect_sizes.png` | the largest per-band effect for every comparison (* = significant) |
| `results/disease_differences.md` / `.csv` / `.json` | every number. The method is in the script's docstring. |
| `logs/run.log` | what the last run printed (the markdown table) |

## Result

Holm correction is applied over all 26 tests. Holm-adjusted p < 0.05 counts as significant.

- **Only one comparison is significant: Chardonnay healthy vs RB only, after SNV.**
  - R² 0.064 vs 0.009 by chance; p = 0.001, Holm 0.026.
  - The largest effect is d = 0.81 at 703 nm, on the red edge. There is also a green effect around 520–620 nm.
- **Suggestive, but not significant after correction:**
  - Vidal healthy vs RB only, SNV: p = 0.008, Holm 0.19, classifier AUC 0.62.
  - Chardonnay RB only vs LR only, SNV: p = 0.025, Holm 0.55, AUC 0.63.
- **Leafroll vs healthy** shows no difference in any variety.
- **Baco** shows no difference for any pair. Its classifier AUCs are 0.43–0.59.
- **Raw spectra** show nothing anywhere: brightness differences between leaves hide the small shape effects.
- **Small groups:** Cab Franc has only 9 healthy leaves and Vidal only 13 LR-only leaves. Their large d values (0.84, 0.72) are what chance produces with groups that small; neither is significant.

The disease signal is far smaller than the variety signal, which separates the varieties with 94% balanced accuracy (see `../variety_logistic_regression/`).
