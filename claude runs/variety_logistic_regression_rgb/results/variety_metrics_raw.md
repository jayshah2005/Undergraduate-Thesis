# Variety prediction with logistic regression (RGB, raw)

Leaf mean at three bands (R 640.0 nm, G 548.5 nm, B 461.0 nm) -> standardise -> multinomial logistic regression, C chosen inside each training split. Same 5 row folds as the 204-band run; every leaf predicted by the model that never saw its row.

**Balanced accuracy: 0.859** (95% CI 0.837-0.880; chance 0.25). Macro-F1: 0.859.

Confusion matrix (rows = true variety, columns = predicted):

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal | total |
|---|---|---|---|---|---|
| **Baco** | 224 | 0 | 13 | 11 | 248 |
| **Cab Franc** | 1 | 197 | 11 | 41 | 250 |
| **Chardonnay** | 13 | 10 | 220 | 7 | 250 |
| **Vidal** | 0 | 22 | 12 | 216 | 250 |

| variety | precision | recall | F1 | n | ROC AUC (one vs rest) |
|---|---|---|---|---|---|
| Baco | 0.941 | 0.903 | 0.922 | 248 | 0.976 |
| Cab Franc | 0.860 | 0.788 | 0.823 | 250 | 0.966 |
| Chardonnay | 0.859 | 0.880 | 0.870 | 250 | 0.969 |
| Vidal | 0.785 | 0.864 | 0.823 | 250 | 0.955 |

## Per fold (held-out row)

| test row | chosen C | test leaves | balanced accuracy |
|---|---|---|---|
| R1 | 10 | 200 | 0.855 |
| R2 | 10 | 200 | 0.850 |
| R3 | 10 | 198 | 0.915 |
| R4 | 10 | 200 | 0.855 |
| R5 | 100 | 200 | 0.820 |

Caveat: 'variety' also carries anything acquired together with it (imaging day, lighting, leaf handling), so a high score shows the spectra separate the varieties, not that cultivar biology alone is the cause.
