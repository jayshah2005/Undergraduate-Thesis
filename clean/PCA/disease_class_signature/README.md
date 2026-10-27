# Is the spectral signature different for healthy / RB only / LR only / both?

Same 998 SNV leaf-mean spectra, coloured by the four disease classes. Fit PCA inside one variety: a pooled plot mostly repeats cultivar (Cab Franc is 96% RB only). Disease p-values shuffle labels within vineyard rows.

```bash
cd clean/PCA/disease_class_signature
../../../.venv/bin/python -u pca_disease_class.py
```

| path | what it is |
|---|---|
| `pca_disease_class.py` | the analysis |
| `REPORT.md` | verdict: not by the 4 classes once variety is held fixed |
| `figures/spectra_by_class_per_variety.png` | class mean ± 1 SD inside each variety |
| `figures/difference_from_healthy.png` | class - healthy (Baco, Chardonnay) |
| `figures/pca_pooled_by_class.png` | all vines; class colour is confounded with variety |
| `figures/pca_within_variety.png` | PCA inside each variety |
| `figures/effect_size.png` | per-band Cohen's d |
| `results/tables.md` | R² / permutation p inside each variety |
| `logs/run.log` | last run |
