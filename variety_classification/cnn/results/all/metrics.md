# Variety: whole-leaf CNN, Full spectrum (204 bands): all metrics

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | 0.874 [0.853, 0.894] | 0.875 [0.855, 0.894] |
| CNN, mean pool | 0.898 [0.880, 0.916] | 0.898 [0.880, 0.916] | 0.897 [0.879, 0.915] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.907 | 0.971 | 0.911 | 0.909 |
| logistic regression | Cab Franc | 0.848 | 0.957 | 0.869 | 0.858 |
| logistic regression | Chardonnay | 0.944 | 0.992 | 0.975 | 0.959 |
| logistic regression | Vidal | 0.796 | 0.912 | 0.751 | 0.773 |
| CNN, mean pool | Baco | 0.931 | 0.976 | 0.928 | 0.930 |
| CNN, mean pool | Cab Franc | 0.856 | 0.977 | 0.926 | 0.890 |
| CNN, mean pool | Chardonnay | 0.992 | 0.945 | 0.858 | 0.920 |
| CNN, mean pool | Vidal | 0.812 | 0.965 | 0.886 | 0.848 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 225 | 0 | 2 | 21 |
| logistic regression | Cab Franc | 0 | 212 | 2 | 36 |
| logistic regression | Chardonnay | 0 | 5 | 236 | 9 |
| logistic regression | Vidal | 22 | 27 | 2 | 199 |
| CNN, mean pool | Baco | 231 | 0 | 12 | 5 |
| CNN, mean pool | Cab Franc | 0 | 214 | 17 | 19 |
| CNN, mean pool | Chardonnay | 0 | 0 | 248 | 2 |
| CNN, mean pool | Vidal | 18 | 17 | 12 | 203 |
