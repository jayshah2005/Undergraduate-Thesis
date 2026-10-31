# Variety logistic regression: `all` bands

**Balanced accuracy: 0.939** (95% CI 0.924-0.952; chance 0.25)

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| **Baco** | 226 | 0 | 1 | 21 |
| **Cab Franc** | 0 | 231 | 4 | 15 |
| **Chardonnay** | 0 | 2 | 244 | 4 |
| **Vidal** | 1 | 13 | 0 | 236 |

| variety | recall | ROC AUC (one vs rest) |
|---|---|---|
| Baco | 0.911 | 0.995 |
| Cab Franc | 0.924 | 0.994 |
| Chardonnay | 0.976 | 0.999 |
| Vidal | 0.944 | 0.981 |

| test row | chosen C | balanced accuracy |
|---|---|---|
| R1 | 0.1 | 0.905 |
| R2 | 0.1 | 0.960 |
| R3 | 1 | 0.990 |
| R4 | 10 | 0.960 |
| R5 | 10 | 0.880 |
