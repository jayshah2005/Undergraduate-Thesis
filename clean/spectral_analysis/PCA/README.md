# PCA of SNV leaf-mean spectra

Each vine is one leaf, represented by the SNV-normalised mean of its pixels over 204 bands (397–1004 nm). SNV removes overall mean reflectance so PCA sees spectral shape. Spectra come from `data.py`.

| folder | question |
|---|---|
| [`variety_signature/`](variety_signature/) | Is the spectral signature different for vines of different variety? |
| [`disease_class_signature/`](disease_class_signature/) | Is the spectral signature different for healthy / RB only / LR only / both? |

Shared math: `pca_common.py`. Run each script from inside its folder:

```bash
../../../../.venv/bin/python -u pca_variety.py
../../../../.venv/bin/python -u pca_disease_class.py
```
