# Variety logistic regression: `all` bands

**Balanced accuracy: 0.874** (95% CI 0.853-0.894; chance 0.25)

| true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|
| **Baco** | 225 | 0 | 2 | 21 |
| **Cab Franc** | 0 | 212 | 2 | 36 |
| **Chardonnay** | 0 | 5 | 236 | 9 |
| **Vidal** | 22 | 27 | 2 | 199 |

| variety | recall | ROC AUC (one vs rest) |
|---|---|---|
| Baco | 0.907 | 0.995 |
| Cab Franc | 0.848 | 0.985 |
| Chardonnay | 0.944 | 0.995 |
| Vidal | 0.796 | 0.961 |

| test row | chosen C | balanced accuracy |
|---|---|---|
| R1 | 0.01 | 0.615 |
| R2 | 10 | 0.975 |
| R3 | 1 | 0.970 |
| R4 | 10 | 0.925 |
| R5 | 1 | 0.885 |
