# Variety prediction with logistic regression (snv spectra)

Leaf mean spectrum -> SNV -> standardise -> multinomial logistic regression, C chosen inside each training split. 5 row folds; every leaf predicted by the model that never saw its row.

**Balanced accuracy: 0.939** (95% CI 0.924-0.954; chance 0.25). Macro-F1: 0.940.

Confusion matrix (rows = true variety, columns = predicted):

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal | total |
|---|---|---|---|---|---|
| **Baco** | 226 | 0 | 1 | 21 | 248 |
| **Cab Franc** | 0 | 231 | 4 | 15 | 250 |
| **Chardonnay** | 0 | 2 | 244 | 4 | 250 |
| **Vidal** | 1 | 13 | 0 | 236 | 250 |

| variety | precision | recall | F1 | n | ROC AUC (one vs rest) |
|---|---|---|---|---|---|
| Baco | 0.996 | 0.911 | 0.952 | 248 | 0.995 |
| Cab Franc | 0.939 | 0.924 | 0.931 | 250 | 0.994 |
| Chardonnay | 0.980 | 0.976 | 0.978 | 250 | 0.999 |
| Vidal | 0.855 | 0.944 | 0.897 | 250 | 0.981 |

## Per fold (held-out row)

| test row | chosen C | test leaves | balanced accuracy |
|---|---|---|---|
| R1 | 0.1 | 200 | 0.905 |
| R2 | 0.1 | 200 | 0.960 |
| R3 | 1 | 198 | 0.990 |
| R4 | 10 | 200 | 0.960 |
| R5 | 10 | 200 | 0.880 |

Caveat: 'variety' also carries anything acquired together with it (imaging day, lighting, leaf handling), so a high score shows the spectra separate the varieties, not that cultivar biology alone is the cause.
