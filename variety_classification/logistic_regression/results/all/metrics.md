# Variety: logistic regression, Full spectrum (204 bands): all metrics

Leaf average of its per-pixel SNV spectra (RGB: normalized rgb) → class-balanced logistic regression; 4 varieties, a leaf is called its most likely one; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.885 [0.870, 0.900] | 0.885 [0.870, 0.900] | 0.886 [0.871, 0.901] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.915 | 0.973 | 0.919 | 0.917 |
| logistic regression | Cab Franc | 0.836 | 0.972 | 0.909 | 0.871 |
| logistic regression | Chardonnay | 0.960 | 0.992 | 0.976 | 0.968 |
| logistic regression | Vidal | 0.828 | 0.909 | 0.753 | 0.789 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 227 | 0 | 0 | 21 |
| logistic regression | Cab Franc | 0 | 209 | 3 | 38 |
| logistic regression | Chardonnay | 0 | 1 | 240 | 9 |
| logistic regression | Vidal | 20 | 20 | 3 | 207 |
