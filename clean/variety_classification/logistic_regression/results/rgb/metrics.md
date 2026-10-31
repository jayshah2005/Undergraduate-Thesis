# Variety logistic regression: `rgb` bands

**Balanced accuracy: 0.500** (95% CI 0.468-0.530; chance 0.25)

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| **Baco** | 99 | 4 | 70 | 75 |
| **Cab Franc** | 37 | 97 | 23 | 93 |
| **Chardonnay** | 49 | 1 | 186 | 14 |
| **Vidal** | 69 | 45 | 19 | 117 |

| variety | recall | ROC AUC (one vs rest) |
|---|---|---|
| Baco | 0.399 | 0.612 |
| Cab Franc | 0.388 | 0.776 |
| Chardonnay | 0.744 | 0.871 |
| Vidal | 0.468 | 0.720 |

| test row | chosen C | balanced accuracy |
|---|---|---|
| R1 | 1 | 0.455 |
| R2 | 0.1 | 0.485 |
| R3 | 0.1 | 0.584 |
| R4 | 0.1 | 0.540 |
| R5 | 1000 | 0.435 |
