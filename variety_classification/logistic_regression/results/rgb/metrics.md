# Variety: logistic regression, RGB (3 bands): all metrics

Leaf average of its per-pixel SNV spectra (RGB: normalized rgb) → class-balanced logistic regression; 4 varieties, a leaf is called its most likely one; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.675 [0.649, 0.702] | 0.675 [0.649, 0.702] | 0.675 [0.649, 0.702] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.520 | 0.828 | 0.500 | 0.510 |
| logistic regression | Cab Franc | 0.624 | 0.953 | 0.817 | 0.707 |
| logistic regression | Chardonnay | 0.804 | 0.888 | 0.705 | 0.751 |
| logistic regression | Vidal | 0.752 | 0.898 | 0.712 | 0.732 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 129 | 21 | 53 | 45 |
| logistic regression | Cab Franc | 42 | 156 | 29 | 23 |
| logistic regression | Chardonnay | 39 | 2 | 201 | 8 |
| logistic regression | Vidal | 48 | 12 | 2 | 188 |
