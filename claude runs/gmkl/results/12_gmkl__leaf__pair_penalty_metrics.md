# 12 GMKL (leaf unit, pair penalty) - out-of-fold results

Settings: unit=leaf, penalty=pair, superpixel px=105/70/53/42/35, tc=30.0, pairs=2000, inner CV folds=10, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the SVM's predicted class; ROC AUC uses its Platt-scaled probabilities. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.353** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 75 | 68 | 44 | 39 | 226 |
| **RB only** | 125 | 308 | 30 | 54 | 517 |
| **LR only** | 53 | 19 | 30 | 37 | 139 |
| **both** | 36 | 22 | 27 | 31 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.559 | 0.501 |
| RB only | 517 | 0.749 | 0.496 |
| LR only | 139 | 0.680 | 0.487 |
| both | 116 | 0.659 | 0.511 |
| **mean** | | 0.662 | 0.499 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 202 | 163 | 218 | 415 | 0.605 | 0.686 | 0.524 |
| leafroll3 | 748 (254) | 333 | 161 | 129 | 125 | 0.583 | 0.598 | 0.484 |
| any_disease | 998 (772) | 75 | 151 | 214 | 558 | 0.527 | 0.559 | 0.501 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.222 | 0.483 | 0.502 | 0.508 | 0.479 | 0.450 | 0.474 | 0.506 |
| Cab Franc | 250 | 0.353 | 0.383 | 0.541 | 0.508 | n/a | n/a | 0.543 | 0.501 |
| Chardonnay | 250 | 0.253 | 0.483 | 0.523 | 0.525 | 0.519 | 0.468 | 0.522 | 0.482 |
| Vidal | 250 | 0.220 | 0.553 | 0.492 | 0.543 | 0.509 | 0.609 | 0.468 | 0.519 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [15, 4, 17, 23]; RB only -> [27, 6, 17, 32]; LR only -> [16, 5, 11, 22]; both -> [15, 8, 11, 19]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [33, 201, 1, 5]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [41, 9, 26, 14]; RB only -> [18, 2, 8, 10]; LR only -> [30, 9, 19, 14]; both -> [18, 5, 15, 12]
- **Vidal**: healthy -> [17, 48, 1, 2]; RB only -> [47, 99, 4, 7]; LR only -> [7, 5, 0, 1]; both -> [3, 8, 1, 0]

## GMKL per fold

eta = learned kernel weights, in the order AS 105px s0.5, AS 105px s1.0, AS 70px s0.5, AS 70px s1.0, AS 53px s0.5, AS 53px s1.0, AS 42px s0.5, AS 42px s1.0, AS 35px s0.5, AS 35px s1.0, EAP.

| fold | C | inner-CV bal. acc. at C | BDR iterations (converged) | negatives clipped | eta |
|---|---|---|---|---|---|
| 0 | 100 | 0.382 | 300 (False) | 5 | 0.00 0.30 0.00 0.25 0.08 0.02 0.00 0.15 0.20 0.00 0.00 |
| 1 | 100 | 0.371 | 300 (False) | 5 | 0.00 0.44 0.00 0.26 0.13 0.00 0.00 0.08 0.06 0.02 0.00 |
| 2 | 10000 | 0.372 | 300 (False) | 6 | 0.00 0.29 0.00 0.23 0.00 0.14 0.00 0.10 0.23 0.00 0.00 |
| 3 | 100 | 0.423 | 300 (False) | 5 | 0.00 0.45 0.00 0.18 0.00 0.17 0.01 0.05 0.14 0.00 0.00 |
| 4 | 10000 | 0.340 | 300 (False) | 6 | 0.00 0.41 0.00 0.22 0.07 0.00 0.00 0.19 0.10 0.00 0.00 |
