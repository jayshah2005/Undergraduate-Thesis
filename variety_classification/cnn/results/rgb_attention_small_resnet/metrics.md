# Variety: whole-leaf CNN, RGB (3 bands): all metrics

4 varieties; a leaf is called its most likely one; CNN = mean probability of 3 seeds; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## RGB (3 bands)

| model | bal. acc. [95% CI] | accuracy [95% CI] | macro F1 [95% CI] |
|---|---|---|---|
| logistic regression | 0.503 [0.472, 0.533] | 0.503 [0.470, 0.535] | 0.493 [0.461, 0.524] |
| small ResNet, attention pool | 0.834 [0.810, 0.856] | 0.834 [0.810, 0.857] | 0.833 [0.809, 0.855] |

Per variety, against the other three:

| model | variety | sensitivity (recall) | specificity | precision | F1 |
|---|---|---|---|---|---|
| logistic regression | Baco | 0.351 | 0.833 | 0.410 | 0.378 |
| logistic regression | Cab Franc | 0.344 | 0.937 | 0.647 | 0.449 |
| logistic regression | Chardonnay | 0.768 | 0.834 | 0.608 | 0.678 |
| logistic regression | Vidal | 0.548 | 0.733 | 0.407 | 0.467 |
| small ResNet, attention pool | Baco | 0.835 | 0.963 | 0.881 | 0.857 |
| small ResNet, attention pool | Cab Franc | 0.852 | 0.975 | 0.918 | 0.884 |
| small ResNet, attention pool | Chardonnay | 0.936 | 0.922 | 0.801 | 0.863 |
| small ResNet, attention pool | Vidal | 0.712 | 0.918 | 0.745 | 0.728 |

Confusion counts (rows = true variety, columns = predicted):

| model | true \ predicted | Baco | Cab Franc | Chardonnay | Vidal |
|---|---|---|---|---|---|
| logistic regression | Baco | 87 | 7 | 76 | 78 |
| logistic regression | Cab Franc | 29 | 86 | 26 | 109 |
| logistic regression | Chardonnay | 43 | 2 | 192 | 13 |
| logistic regression | Vidal | 53 | 38 | 22 | 137 |
| small ResNet, attention pool | Baco | 207 | 0 | 8 | 33 |
| small ResNet, attention pool | Cab Franc | 7 | 213 | 10 | 20 |
| small ResNet, attention pool | Chardonnay | 7 | 1 | 234 | 8 |
| small ResNet, attention pool | Vidal | 14 | 18 | 40 | 178 |
