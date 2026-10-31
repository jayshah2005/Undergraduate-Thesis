# 12 GMKL (leaf unit, pair penalty) - out-of-fold results

Settings: unit=leaf, penalty=pair, superpixel px=105/70/53/42/35, tc=30.0, pairs=2000, inner CV folds=10, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the SVM's predicted class; ROC AUC uses its Platt-scaled probabilities. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.381** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 66 | 62 | 47 | 51 | 226 |
| **RB only** | 102 | 315 | 36 | 64 | 517 |
| **LR only** | 50 | 8 | 42 | 39 | 139 |
| **both** | 36 | 16 | 27 | 37 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.571 | 0.503 |
| RB only | 517 | 0.776 | 0.515 |
| LR only | 139 | 0.700 | 0.492 |
| both | 116 | 0.648 | 0.467 |
| **mean** | | 0.674 | 0.494 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 205 | 160 | 201 | 432 | 0.622 | 0.711 | 0.531 |
| leafroll3 | 748 (254) | 300 | 194 | 109 | 145 | 0.589 | 0.603 | 0.462 |
| any_disease | 998 (772) | 66 | 160 | 188 | 584 | 0.524 | 0.571 | 0.503 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.224 | 0.484 | 0.495 | 0.488 | 0.459 | 0.456 | 0.458 | 0.470 |
| Cab Franc | 250 | 0.352 | 0.364 | 0.534 | 0.455 | n/a | n/a | 0.536 | 0.476 |
| Chardonnay | 250 | 0.282 | 0.468 | 0.568 | 0.564 | 0.521 | 0.417 | 0.523 | 0.512 |
| Vidal | 250 | 0.233 | 0.566 | 0.511 | 0.561 | 0.482 | 0.593 | 0.495 | 0.525 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [10, 3, 17, 29]; RB only -> [18, 1, 23, 40]; LR only -> [15, 1, 13, 25]; both -> [15, 2, 11, 25]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [36, 200, 1, 3]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [39, 3, 30, 18]; RB only -> [16, 2, 9, 11]; LR only -> [29, 0, 29, 14]; both -> [17, 6, 15, 12]
- **Vidal**: healthy -> [15, 49, 0, 4]; RB only -> [32, 112, 3, 10]; LR only -> [6, 7, 0, 0]; both -> [4, 7, 1, 0]

## GMKL per fold

eta = learned kernel weights, in the order AS 105px s0.5, AS 105px s1.0, AS 70px s0.5, AS 70px s1.0, AS 53px s0.5, AS 53px s1.0, AS 42px s0.5, AS 42px s1.0, AS 35px s0.5, AS 35px s1.0, EAP.

| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |
|---|---|---|---|---|---|
| 0 | 100 | 0.394 | 285 (True) | 4 | 0.00 0.35 0.00 0.21 0.03 0.14 0.12 0.11 0.04 0.00 0.00 |
| 1 | 100 | 0.359 | 300 (False) | 6 | 0.00 0.26 0.00 0.26 0.00 0.17 0.00 0.16 0.16 0.00 0.00 |
| 2 | 100 | 0.362 | 300 (False) | 6 | 0.00 0.27 0.00 0.24 0.00 0.15 0.00 0.14 0.20 0.00 0.00 |
| 3 | 100 | 0.415 | 300 (False) | 4 | 0.00 0.36 0.00 0.18 0.03 0.15 0.06 0.14 0.08 0.00 0.00 |
| 4 | 1000 | 0.353 | 300 (False) | 4 | 0.00 0.33 0.00 0.26 0.04 0.11 0.01 0.09 0.15 0.00 0.00 |
