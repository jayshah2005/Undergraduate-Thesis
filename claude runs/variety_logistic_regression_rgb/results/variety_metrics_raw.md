# Variety prediction with logistic regression (RGB, raw)

Leaf mean at three bands (R 639.4 nm, G 550.7 nm, B 459.1 nm) -> standardise -> multinomial logistic regression, C chosen inside each training split. Same 5 row folds as the 204-band run; every leaf predicted by the model that never saw its row.

**Balanced accuracy: 0.848** (95% CI 0.825-0.870; chance 0.25). Macro-F1: 0.848.

Confusion matrix (rows = true variety, columns = predicted):

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal | total |
|---|---|---|---|---|---|
| **Baco** | 225 | 0 | 15 | 8 | 248 |
| **Cab Franc** | 0 | 190 | 12 | 48 | 250 |
| **Chardonnay** | 13 | 7 | 218 | 12 | 250 |
| **Vidal** | 0 | 23 | 14 | 213 | 250 |

| variety | precision | recall | F1 | n | ROC AUC (one vs rest) |
|---|---|---|---|---|---|
| Baco | 0.945 | 0.907 | 0.926 | 248 | 0.981 |
| Cab Franc | 0.864 | 0.760 | 0.809 | 250 | 0.964 |
| Chardonnay | 0.842 | 0.872 | 0.857 | 250 | 0.969 |
| Vidal | 0.758 | 0.852 | 0.802 | 250 | 0.953 |

## Per fold (held-out row)

| test row | chosen C | test leaves | balanced accuracy |
|---|---|---|---|
| R1 | 10 | 200 | 0.870 |
| R2 | 10 | 200 | 0.835 |
| R3 | 10 | 198 | 0.895 |
| R4 | 10 | 200 | 0.810 |
| R5 | 100 | 200 | 0.830 |

Caveat: 'variety' also carries anything acquired together with it (imaging day, lighting, leaf handling), so a high score shows the spectra separate the varieties, not that cultivar biology alone is the cause.
