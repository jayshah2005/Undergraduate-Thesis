# Is the spectral signature different for vines of different variety?

PCA of the 998 SNV leaf-mean spectra, coloured by Baco / Cab Franc / Chardonnay / Vidal.

```bash
cd clean/PCA/variety_signature
../../../.venv/bin/python -u pca_variety.py
```

| path | what it is |
|---|---|
| `pca_variety.py` | the analysis |
| `REPORT.md` | verdict: yes, by variety |
| `figures/spectra_by_variety.png` | every vine's SNV curve + variety means |
| `figures/difference_from_grand_mean.png` | variety mean - grand mean |
| `figures/pca.png` | PC1-2 and PC2-3 |
| `results/tables.md` | R² of variety and eta² on each PC |
| `logs/run.log` | last run |
