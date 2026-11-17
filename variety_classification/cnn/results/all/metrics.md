# Variety: whole-leaf CNN, Full spectrum (204 bands): all metrics

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Full spectrum (204 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.874 [0.853, 0.894] | 0.874 [0.853, 0.894] | 0.875 [0.855, 0.894] |
| CNN, mean pool | 0.898 [0.880, 0.916] | 0.898 [0.880, 0.916] | 0.897 [0.879, 0.915] |
| CNN, attention pool | 0.895 [0.876, 0.914] | 0.895 [0.877, 0.914] | 0.895 [0.876, 0.913] |
| small ResNet, mean pool | 0.907 [0.890, 0.924] | 0.907 [0.890, 0.924] | 0.907 [0.891, 0.925] |
| small ResNet, attention pool | 0.902 [0.884, 0.920] | 0.902 [0.884, 0.921] | 0.901 [0.882, 0.919] |

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
| CNN, attention pool | Baco | 0.931 | 0.995 | 0.983 | 0.957 |
| CNN, attention pool | Cab Franc | 0.876 | 0.937 | 0.823 | 0.849 |
| CNN, attention pool | Chardonnay | 0.976 | 0.980 | 0.942 | 0.959 |
| CNN, attention pool | Vidal | 0.796 | 0.948 | 0.836 | 0.816 |
| small ResNet, mean pool | Baco | 0.988 | 1.000 | 1.000 | 0.994 |
| small ResNet, mean pool | Cab Franc | 0.836 | 0.936 | 0.813 | 0.824 |
| small ResNet, mean pool | Chardonnay | 0.996 | 0.999 | 0.996 | 0.996 |
| small ResNet, mean pool | Vidal | 0.808 | 0.941 | 0.821 | 0.815 |
| small ResNet, attention pool | Baco | 0.972 | 0.955 | 0.876 | 0.922 |
| small ResNet, attention pool | Cab Franc | 0.836 | 0.981 | 0.937 | 0.884 |
| small ResNet, attention pool | Chardonnay | 0.996 | 0.988 | 0.965 | 0.980 |
| small ResNet, attention pool | Vidal | 0.804 | 0.945 | 0.831 | 0.817 |

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
| CNN, attention pool | Baco | 231 | 0 | 13 | 4 |
| CNN, attention pool | Cab Franc | 0 | 219 | 1 | 30 |
| CNN, attention pool | Chardonnay | 0 | 1 | 244 | 5 |
| CNN, attention pool | Vidal | 4 | 46 | 1 | 199 |
| small ResNet, mean pool | Baco | 245 | 0 | 0 | 3 |
| small ResNet, mean pool | Cab Franc | 0 | 209 | 1 | 40 |
| small ResNet, mean pool | Chardonnay | 0 | 0 | 249 | 1 |
| small ResNet, mean pool | Vidal | 0 | 48 | 0 | 202 |
| small ResNet, attention pool | Baco | 241 | 0 | 4 | 3 |
| small ResNet, attention pool | Cab Franc | 0 | 209 | 4 | 37 |
| small ResNet, attention pool | Chardonnay | 0 | 0 | 249 | 1 |
| small ResNet, attention pool | Vidal | 34 | 14 | 1 | 201 |
