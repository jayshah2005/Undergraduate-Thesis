# Variety: whole-leaf CNN, RGB (3 bands): all metrics, trained without augmentation

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | 0.503 [0.470, 0.535] | 0.493 [0.461, 0.524] |
| CNN, mean pool | 0.928 [0.911, 0.943] | 0.928 [0.912, 0.944] | 0.929 [0.912, 0.944] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.351 | 0.833 | 0.410 | 0.378 |
| logistic regression | Cab Franc | 0.344 | 0.937 | 0.647 | 0.449 |
| logistic regression | Chardonnay | 0.768 | 0.834 | 0.608 | 0.678 |
| logistic regression | Vidal | 0.548 | 0.733 | 0.407 | 0.467 |
| CNN, mean pool | Baco | 0.907 | 0.996 | 0.987 | 0.945 |
| CNN, mean pool | Cab Franc | 0.908 | 0.979 | 0.934 | 0.921 |
| CNN, mean pool | Chardonnay | 0.980 | 0.988 | 0.965 | 0.972 |
| CNN, mean pool | Vidal | 0.916 | 0.941 | 0.839 | 0.876 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 87 | 7 | 76 | 78 |
| logistic regression | Cab Franc | 29 | 86 | 26 | 109 |
| logistic regression | Chardonnay | 43 | 2 | 192 | 13 |
| logistic regression | Vidal | 53 | 38 | 22 | 137 |
| CNN, mean pool | Baco | 225 | 0 | 1 | 22 |
| CNN, mean pool | Cab Franc | 1 | 227 | 4 | 18 |
| CNN, mean pool | Chardonnay | 0 | 1 | 245 | 4 |
| CNN, mean pool | Vidal | 2 | 15 | 4 | 229 |
