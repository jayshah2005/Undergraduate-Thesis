# Spectral analysis

What the leaf spectra look like, with no prediction model.

| path | what it is |
|---|---|
| [`PCA/`](PCA/) | do SNV leaf-mean spectra differ by variety, and by disease class? |
| `normalization_figure/plot_pipeline.py` | poster schematic: cube → 1 × 204 mean → SNV → stack (a diagram; rows coloured from a made-up spectrum) |

```bash
cd clean/spectral_analysis/normalization_figure
../../../.venv/bin/python -u plot_pipeline.py    # figures/pipeline.png (300 dpi)
```
