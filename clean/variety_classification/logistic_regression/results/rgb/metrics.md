# Variety logistic regression: `rgb` bands

**Balanced accuracy: 0.503** (95% CI 0.472-0.533; chance 0.25)

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| **Baco** | 87 | 7 | 76 | 78 |
| **Cab Franc** | 29 | 86 | 26 | 109 |
| **Chardonnay** | 43 | 2 | 192 | 13 |
| **Vidal** | 53 | 38 | 22 | 137 |

| variety | recall | ROC AUC (one vs rest) |
|---|---|---|
| Baco | 0.351 | 0.599 |
| Cab Franc | 0.344 | 0.774 |
| Chardonnay | 0.768 | 0.870 |
| Vidal | 0.548 | 0.719 |

| test row | chosen C | balanced accuracy |
|---|---|---|
| R1 | 0.1 | 0.420 |
| R2 | 1 | 0.500 |
| R3 | 1000 | 0.599 |
| R4 | 0.1 | 0.520 |
| R5 | 0.1 | 0.475 |
