# 12 GMKL (pixel unit, full penalty) - out-of-fold results

Settings: unit=pixel, penalty=full, superpixel px=105/70/53/42/35, tc=30.0, pairs=2000, inner CV folds=10, folds=0,1,2,3,4, train pixels per class=1000

Every leaf is predicted by the model of the fold that held out its row. Decisions = the majority vote of the SVM's predicted classes over the leaf's pixels; ROC AUC uses the mean of the pixels' Platt-scaled probabilities. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.419** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 60 | 67 | 53 | 46 | 226 |
| **RB only** | 48 | 366 | 46 | 57 | 517 |
| **LR only** | 46 | 10 | 45 | 38 | 139 |
| **both** | 22 | 14 | 36 | 44 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.597 | 0.531 |
| RB only | 517 | 0.806 | 0.541 |
| LR only | 139 | 0.744 | 0.527 |
| both | 116 | 0.721 | 0.532 |
| **mean** | | 0.717 | 0.533 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 204 | 161 | 152 | 481 | 0.659 | 0.749 | 0.563 |
| leafroll3 | 748 (254) | 292 | 202 | 91 | 163 | 0.616 | 0.649 | 0.488 |
| any_disease | 998 (772) | 60 | 166 | 116 | 656 | 0.558 | 0.597 | 0.531 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.256 | 0.479 | 0.495 | 0.479 | 0.480 | 0.447 | 0.502 | 0.391 |
| Cab Franc | 250 | 0.387 | 0.518 | 0.580 | 0.619 | n/a | n/a | 0.580 | 0.603 |
| Chardonnay | 250 | 0.275 | 0.546 | 0.584 | 0.618 | 0.504 | 0.482 | 0.534 | 0.579 |
| Vidal | 250 | 0.258 | 0.577 | 0.521 | 0.594 | 0.500 | 0.612 | 0.508 | 0.583 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [3, 1, 18, 37]; RB only -> [1, 0, 33, 48]; LR only -> [4, 0, 20, 30]; both -> [4, 0, 17, 32]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [15, 225, 0, 0]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [46, 0, 35, 9]; RB only -> [16, 0, 13, 9]; LR only -> [39, 0, 25, 8]; both -> [16, 3, 19, 12]
- **Vidal**: healthy -> [9, 59, 0, 0]; RB only -> [16, 141, 0, 0]; LR only -> [3, 10, 0, 0]; both -> [2, 10, 0, 0]

## GMKL per fold

eta = learned kernel weights, in the order AS 105px s0.5, AS 105px s1.0, AS 70px s0.5, AS 70px s1.0, AS 53px s0.5, AS 53px s1.0, AS 42px s0.5, AS 42px s1.0, AS 35px s0.5, AS 35px s1.0, EAP.

| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |
|---|---|---|---|---|---|
| 0 | 100 | 0.392 | 300 (False) | 6 | 0.00 0.23 0.00 0.22 0.00 0.18 0.00 0.16 0.00 0.21 0.00 |
| 1 | 100 | 0.409 | 300 (False) | 6 | 0.00 0.26 0.00 0.16 0.00 0.15 0.00 0.22 0.00 0.20 0.00 |
| 2 | 100 | 0.395 | 300 (False) | 6 | 0.00 0.24 0.00 0.20 0.00 0.19 0.00 0.18 0.00 0.19 0.00 |
| 3 | 100 | 0.413 | 300 (False) | 6 | 0.00 0.20 0.00 0.21 0.00 0.23 0.00 0.18 0.00 0.18 0.00 |
| 4 | 100 | 0.397 | 300 (False) | 6 | 0.00 0.24 0.00 0.16 0.00 0.20 0.00 0.19 0.00 0.20 0.00 |
