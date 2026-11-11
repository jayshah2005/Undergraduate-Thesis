# Variety: logistic regression, RGB (3 bands): all metrics

Leaf mean spectrum → SNV → logistic regression; 4 varieties, a leaf is called its most likely one; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | 0.503 [0.470, 0.535] | 0.493 [0.461, 0.524] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.351 | 0.833 | 0.410 | 0.378 |
| logistic regression | Cab Franc | 0.344 | 0.937 | 0.647 | 0.449 |
| logistic regression | Chardonnay | 0.768 | 0.834 | 0.608 | 0.678 |
| logistic regression | Vidal | 0.548 | 0.733 | 0.407 | 0.467 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 87 | 7 | 76 | 78 |
| logistic regression | Cab Franc | 29 | 86 | 26 | 109 |
| logistic regression | Chardonnay | 43 | 2 | 192 | 13 |
| logistic regression | Vidal | 53 | 38 | 22 | 137 |
