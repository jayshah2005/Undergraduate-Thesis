# Variety prediction with logistic regression (RGB, snv)

Leaf mean at three bands (R 639.4 nm, G 550.7 nm, B 459.1 nm) -> SNV -> standardise -> multinomial logistic regression, C chosen inside each training split. Same 5 row folds as the 204-band run; every leaf predicted by the model that never saw its row.

**Balanced accuracy: 0.506** (95% CI 0.477-0.536; chance 0.25). Macro-F1: 0.504.

Confusion matrix (rows = true variety, columns = predicted):

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal | total |
|---|---|---|---|---|---|
| **Baco** | 91 | 4 | 78 | 75 | 248 |
| **Cab Franc** | 23 | 116 | 23 | 88 | 250 |
| **Chardonnay** | 52 | 2 | 180 | 16 | 250 |
| **Vidal** | 62 | 48 | 22 | 118 | 250 |

| variety | precision | recall | F1 | n | ROC AUC (one vs rest) |
|---|---|---|---|---|---|
| Baco | 0.399 | 0.367 | 0.382 | 248 | 0.628 |
| Cab Franc | 0.682 | 0.464 | 0.552 | 250 | 0.804 |
| Chardonnay | 0.594 | 0.720 | 0.651 | 250 | 0.862 |
| Vidal | 0.397 | 0.472 | 0.431 | 250 | 0.705 |

## Per fold (held-out row)

| test row | chosen C | test leaves | balanced accuracy |
|---|---|---|---|
| R1 | 0.1 | 200 | 0.420 |
| R2 | 0.1 | 200 | 0.480 |
| R3 | 0.1 | 198 | 0.599 |
| R4 | 0.1 | 200 | 0.585 |
| R5 | 1000 | 200 | 0.445 |

Caveat: 'variety' also carries anything acquired together with it (imaging day, lighting, leaf handling), so a high score shows the spectra separate the varieties, not that cultivar biology alone is the cause.
