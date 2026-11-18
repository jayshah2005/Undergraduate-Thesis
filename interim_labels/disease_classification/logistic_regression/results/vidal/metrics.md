# Disease: logistic regression on Vidal, full spectrum: all metrics

A leaf counts as positive (diseased) at P ≥ 0.5; 95% bootstrap intervals. Sensitivity (= recall): share of positive leaves called positive. Specificity: share of negative leaves called negative. Precision: share of positive calls that are right. F1: harmonic mean of precision and sensitivity. Balanced accuracy: mean of sensitivity and specificity (for variety, mean sensitivity over the 4 varieties). Balanced accuracy alone, with folds and differences to logistic regression: `final_report.md`.

## Leafroll (25 + / 225 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.544 [0.491, 0.596] | 0.660 [0.616, 0.700] | 0.400 [0.320, 0.480] | 0.689 [0.640, 0.733] | 0.125 [0.096, 0.157] | 0.190 [0.147, 0.235] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 10 | 15 | 70 | 155 |

## Red blotch (169 + / 81 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.552 [0.494, 0.610] | 0.612 [0.560, 0.664] | 0.722 [0.651, 0.787] | 0.383 [0.284, 0.481] | 0.709 [0.672, 0.746] | 0.716 [0.669, 0.758] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 122 | 47 | 50 | 31 |

## Both diseases (12 + / 238 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.479 [0.424, 0.562] | 0.836 [0.800, 0.872] | 0.083 [0.000, 0.250] | 0.874 [0.836, 0.912] | 0.032 [0.000, 0.094] | 0.047 [0.000, 0.133] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 1 | 11 | 30 | 208 |

## Any disease vs healthy (182 + / 68 −)

| model | bal. acc. [95% CI] | accuracy [95% CI] | sensitivity (recall) [95% CI] | specificity [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|---|
| logistic regression | 0.577 [0.516, 0.638] | 0.652 [0.600, 0.704] | 0.742 [0.681, 0.802] | 0.412 [0.309, 0.515] | 0.771 [0.737, 0.807] | 0.756 [0.714, 0.798] |

Confusion counts (positive = called diseased at P ≥ 0.5):

| model | TP | FN | FP | TN |
|---|---|---|---|---|
| logistic regression | 135 | 47 | 40 | 28 |
