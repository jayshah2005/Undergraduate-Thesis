# Verdict: yes, the SNV spectrum differs by variety

998 vines, one SNV-normalised mean spectrum each (204 bands). Mean reflectance is removed, so the remaining variation is spectral shape.

R² of variety is 0.510 (chance 0.003, p = 0.001).

- Variety owns PC1-3 (η² 0.59 / 0.51 / 0.38). Those three PCs hold 47.9% + 34.2% + 13.1% of the SNV variance.
- PC1 is no longer mean reflectance (r = +0.17, vs +1.00 on raw spectra).
- Four clouds in `figures/pca.png`.

Caveat: "variety" includes whatever was acquired with that cultivar (imaging day, handling). The split is real in this dataset. It is not a proof of cultivar biology alone.

Numbers: `results/tables.md`.
