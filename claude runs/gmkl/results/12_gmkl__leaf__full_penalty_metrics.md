# 12 GMKL (leaf unit, full penalty) - out-of-fold results

Settings: unit=leaf, penalty=full, superpixel px=105/70/53/42/35, tc=30.0, pairs=2000, inner CV folds=10, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the SVM's predicted class; ROC AUC uses its Platt-scaled probabilities. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.383** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 73 | 57 | 50 | 46 | 226 |
| **RB only** | 100 | 320 | 35 | 62 | 517 |
| **LR only** | 52 | 13 | 40 | 34 | 139 |
| **both** | 36 | 18 | 27 | 35 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.567 | 0.499 |
| RB only | 517 | 0.771 | 0.519 |
| LR only | 139 | 0.692 | 0.493 |
| both | 116 | 0.647 | 0.477 |
| **mean** | | 0.669 | 0.497 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 215 | 150 | 198 | 435 | 0.638 | 0.706 | 0.537 |
| leafroll3 | 748 (254) | 305 | 189 | 118 | 136 | 0.576 | 0.603 | 0.462 |
| any_disease | 998 (772) | 73 | 153 | 188 | 584 | 0.540 | 0.567 | 0.499 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.234 | 0.494 | 0.527 | 0.503 | 0.451 | 0.454 | 0.478 | 0.495 |
| Cab Franc | 250 | 0.356 | 0.352 | 0.541 | 0.459 | n/a | n/a | 0.543 | 0.441 |
| Chardonnay | 250 | 0.275 | 0.464 | 0.559 | 0.559 | 0.513 | 0.417 | 0.526 | 0.489 |
| Vidal | 250 | 0.246 | 0.573 | 0.545 | 0.567 | 0.478 | 0.600 | 0.520 | 0.525 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [14, 3, 18, 24]; RB only -> [20, 2, 22, 38]; LR only -> [18, 3, 13, 20]; both -> [15, 4, 11, 23]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [33, 203, 1, 3]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [39, 3, 30, 18]; RB only -> [16, 2, 9, 11]; LR only -> [28, 3, 27, 14]; both -> [17, 6, 15, 12]
- **Vidal**: healthy -> [18, 44, 2, 4]; RB only -> [31, 113, 3, 10]; LR only -> [6, 7, 0, 0]; both -> [4, 7, 1, 0]

## GMKL per fold

eta = learned kernel weights, in the order AS 105px s0.5, AS 105px s1.0, AS 70px s0.5, AS 70px s1.0, AS 53px s0.5, AS 53px s1.0, AS 42px s0.5, AS 42px s1.0, AS 35px s0.5, AS 35px s1.0, EAP.

| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |
|---|---|---|---|---|---|
| 0 | 100 | 0.395 | 285 (True) | 4 | 0.00 0.35 0.00 0.21 0.02 0.14 0.11 0.11 0.05 0.00 0.00 |
| 1 | 100 | 0.359 | 300 (False) | 6 | 0.00 0.25 0.00 0.26 0.00 0.17 0.00 0.16 0.16 0.00 0.00 |
| 2 | 100 | 0.362 | 300 (False) | 6 | 0.00 0.27 0.00 0.24 0.00 0.15 0.00 0.14 0.20 0.00 0.00 |
| 3 | 100 | 0.416 | 300 (False) | 4 | 0.00 0.36 0.00 0.19 0.03 0.15 0.05 0.13 0.08 0.00 0.00 |
| 4 | 10000 | 0.349 | 300 (False) | 4 | 0.00 0.33 0.00 0.26 0.04 0.11 0.02 0.09 0.15 0.00 0.00 |
