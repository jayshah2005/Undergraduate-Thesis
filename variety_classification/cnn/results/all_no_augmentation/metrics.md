# Variety: whole-leaf CNN, Full spectrum (204 bands): all metrics, trained without augmentation

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | 0.874 [0.853, 0.894] | 0.875 [0.855, 0.894] |
| CNN, mean pool | 0.914 [0.898, 0.931] | 0.914 [0.897, 0.931] | 0.914 [0.897, 0.930] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.907 | 0.971 | 0.911 | 0.909 |
| logistic regression | Cab Franc | 0.848 | 0.957 | 0.869 | 0.858 |
| logistic regression | Chardonnay | 0.944 | 0.992 | 0.975 | 0.959 |
| logistic regression | Vidal | 0.796 | 0.912 | 0.751 | 0.773 |
| CNN, mean pool | Baco | 0.972 | 1.000 | 1.000 | 0.986 |
| CNN, mean pool | Cab Franc | 0.788 | 0.976 | 0.916 | 0.847 |
| CNN, mean pool | Chardonnay | 0.984 | 0.980 | 0.943 | 0.963 |
| CNN, mean pool | Vidal | 0.912 | 0.929 | 0.811 | 0.859 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 225 | 0 | 2 | 21 |
| logistic regression | Cab Franc | 0 | 212 | 2 | 36 |
| logistic regression | Chardonnay | 0 | 5 | 236 | 9 |
| logistic regression | Vidal | 22 | 27 | 2 | 199 |
| CNN, mean pool | Baco | 241 | 0 | 0 | 7 |
| CNN, mean pool | Cab Franc | 0 | 197 | 11 | 42 |
| CNN, mean pool | Chardonnay | 0 | 0 | 246 | 4 |
| CNN, mean pool | Vidal | 0 | 18 | 4 | 228 |
