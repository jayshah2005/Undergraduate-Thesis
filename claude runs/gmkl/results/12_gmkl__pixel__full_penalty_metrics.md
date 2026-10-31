# 12 GMKL (pixel unit, full penalty) - out-of-fold results

Settings: unit=pixel, penalty=full, superpixel px=105/70/53/42/35, tc=30.0, pairs=2000, inner CV folds=10, folds=0,1,2,3,4, train pixels per class=1000

Every leaf is predicted by the model of the fold that held out its row. Decisions = the majority vote of the SVM's predicted classes over the leaf's pixels; ROC AUC uses the mean of the pixels' Platt-scaled probabilities. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.422** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 55 | 69 | 58 | 44 | 226 |
| **RB only** | 43 | 367 | 50 | 57 | 517 |
| **LR only** | 43 | 11 | 48 | 37 | 139 |
| **both** | 20 | 14 | 37 | 45 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.588 | 0.509 |
| RB only | 517 | 0.802 | 0.529 |
| LR only | 139 | 0.745 | 0.526 |
| both | 116 | 0.719 | 0.531 |
| **mean** | | 0.713 | 0.524 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 204 | 161 | 150 | 483 | 0.661 | 0.749 | 0.557 |
| leafroll3 | 748 (254) | 285 | 209 | 87 | 167 | 0.617 | 0.643 | 0.480 |
| any_disease | 998 (772) | 55 | 171 | 106 | 666 | 0.553 | 0.588 | 0.509 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.261 | 0.475 | 0.500 | 0.483 | 0.479 | 0.432 | 0.496 | 0.379 |
| Cab Franc | 250 | 0.388 | 0.452 | 0.582 | 0.610 | n/a | n/a | 0.582 | 0.575 |
| Chardonnay | 250 | 0.277 | 0.532 | 0.579 | 0.612 | 0.505 | 0.470 | 0.539 | 0.548 |
| Vidal | 250 | 0.250 | 0.574 | 0.511 | 0.576 | 0.493 | 0.631 | 0.505 | 0.562 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [3, 1, 18, 37]; RB only -> [2, 0, 33, 47]; LR only -> [5, 0, 20, 29]; both -> [4, 0, 16, 33]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [14, 226, 0, 0]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [43, 0, 40, 7]; RB only -> [14, 0, 17, 7]; LR only -> [36, 0, 28, 8]; both -> [14, 3, 21, 12]
- **Vidal**: healthy -> [7, 61, 0, 0]; RB only -> [13, 141, 0, 3]; LR only -> [2, 11, 0, 0]; both -> [2, 10, 0, 0]

## GMKL per fold

eta = learned kernel weights, in the order AS 105px s0.5, AS 105px s1.0, AS 70px s0.5, AS 70px s1.0, AS 53px s0.5, AS 53px s1.0, AS 42px s0.5, AS 42px s1.0, AS 35px s0.5, AS 35px s1.0, EAP.

| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |
|---|---|---|---|---|---|
| 0 | 100 | 0.393 | 300 (False) | 6 | 0.00 0.26 0.00 0.19 0.00 0.20 0.00 0.20 0.00 0.15 0.00 |
| 1 | 100 | 0.411 | 300 (False) | 6 | 0.00 0.27 0.00 0.20 0.00 0.18 0.00 0.18 0.00 0.17 0.00 |
| 2 | 100 | 0.381 | 300 (False) | 6 | 0.00 0.26 0.00 0.18 0.00 0.20 0.00 0.18 0.00 0.18 0.00 |
| 3 | 100 | 0.409 | 300 (False) | 6 | 0.00 0.27 0.00 0.18 0.00 0.19 0.00 0.20 0.00 0.16 0.00 |
| 4 | 100 | 0.400 | 300 (False) | 6 | 0.00 0.25 0.00 0.20 0.00 0.20 0.00 0.17 0.00 0.19 0.00 |
