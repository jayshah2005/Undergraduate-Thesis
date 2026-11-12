# Variety: whole-leaf CNN, RGB (3 bands): all metrics

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | 0.503 [0.470, 0.535] | 0.493 [0.461, 0.524] |
| CNN, mean pool | 0.824 [0.799, 0.846] | 0.824 [0.800, 0.848] | 0.823 [0.797, 0.845] |
| small ResNet, mean pool | 0.885 [0.865, 0.904] | 0.885 [0.865, 0.905] | 0.885 [0.866, 0.905] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.351 | 0.833 | 0.410 | 0.378 |
| logistic regression | Cab Franc | 0.344 | 0.937 | 0.647 | 0.449 |
| logistic regression | Chardonnay | 0.768 | 0.834 | 0.608 | 0.678 |
| logistic regression | Vidal | 0.548 | 0.733 | 0.407 | 0.467 |
| CNN, mean pool | Baco | 0.794 | 0.960 | 0.868 | 0.829 |
| CNN, mean pool | Cab Franc | 0.880 | 0.967 | 0.898 | 0.889 |
| CNN, mean pool | Chardonnay | 0.924 | 0.925 | 0.805 | 0.860 |
| CNN, mean pool | Vidal | 0.696 | 0.913 | 0.728 | 0.712 |
| small ResNet, mean pool | Baco | 0.871 | 0.988 | 0.960 | 0.913 |
| small ResNet, mean pool | Cab Franc | 0.900 | 0.988 | 0.962 | 0.930 |
| small ResNet, mean pool | Chardonnay | 0.960 | 0.922 | 0.805 | 0.876 |
| small ResNet, mean pool | Vidal | 0.808 | 0.948 | 0.838 | 0.823 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 87 | 7 | 76 | 78 |
| logistic regression | Cab Franc | 29 | 86 | 26 | 109 |
| logistic regression | Chardonnay | 43 | 2 | 192 | 13 |
| logistic regression | Vidal | 53 | 38 | 22 | 137 |
| CNN, mean pool | Baco | 197 | 0 | 14 | 37 |
| CNN, mean pool | Cab Franc | 2 | 220 | 11 | 17 |
| CNN, mean pool | Chardonnay | 8 | 0 | 231 | 11 |
| CNN, mean pool | Vidal | 20 | 25 | 31 | 174 |
| small ResNet, mean pool | Baco | 216 | 0 | 7 | 25 |
| small ResNet, mean pool | Cab Franc | 1 | 225 | 17 | 7 |
| small ResNet, mean pool | Chardonnay | 3 | 0 | 240 | 7 |
| small ResNet, mean pool | Vidal | 5 | 9 | 34 | 202 |
