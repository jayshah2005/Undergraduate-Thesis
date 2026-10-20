# PMMFNet controlled experiment A2 - A1 + scaler fitted on training leaves only

Settings: experiment=A2, scaler=train, train_shuffle=True, eval_shuffle=False, objective=patch, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.402** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 43 | 54 | 94 | 35 | 226 |
| **RB only** | 88 | 325 | 50 | 54 | 517 |
| **LR only** | 21 | 14 | 69 | 35 | 139 |
| **both** | 18 | 17 | 47 | 34 | 116 |

Macro-F1: **0.378**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.253 | 0.190 | 0.217 | 226 |
| RB only | 0.793 | 0.629 | 0.701 | 517 |
| LR only | 0.265 | 0.496 | 0.346 | 139 |
| both | 0.215 | 0.293 | 0.248 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.548 | 0.501 |
| RB only | 517 | 0.805 | 0.545 |
| LR only | 139 | 0.725 | 0.515 |
| both | 116 | 0.662 | 0.525 |
| **mean** | | 0.685 | 0.522 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 227 | 138 | 203 | 430 | 0.651 | 0.726 | 0.583 |
| leafroll3 | 748 (254) | 265 | 229 | 69 | 185 | 0.632 | 0.633 | 0.473 |
| any_disease | 998 (772) | 43 | 183 | 127 | 645 | 0.513 | 0.548 | 0.501 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.270 | 0.529 | 0.561 | 0.579 | 0.493 | 0.472 | 0.459 | 0.429 |
| Cab Franc | 250 | 0.385 | 0.512 | 0.582 | 0.613 | n/a | n/a | 0.586 | 0.585 |
| Chardonnay | 250 | 0.267 | 0.518 | 0.550 | 0.607 | 0.543 | 0.482 | 0.518 | 0.509 |
| Vidal | 250 | 0.245 | 0.489 | 0.555 | 0.558 | 0.469 | 0.456 | 0.543 | 0.539 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [8, 0, 35, 16]; RB only -> [19, 4, 33, 26]; LR only -> [12, 1, 29, 12]; both -> [10, 2, 22, 19]
- **Cab Franc**: healthy -> [3, 6, 0, 0]; RB only -> [39, 197, 2, 2]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [19, 0, 54, 17]; RB only -> [13, 0, 13, 12]; LR only -> [8, 1, 40, 23]; both -> [7, 4, 24, 15]
- **Vidal**: healthy -> [13, 48, 5, 2]; RB only -> [17, 124, 2, 14]; LR only -> [1, 12, 0, 0]; both -> [1, 10, 1, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 5 | 10 | 0.343 | 1.4886 | 0.326 | 8.3 |
| 1 | 2 | [0.0073, 2.8844] | 5 | 10 | 0.427 | 1.2744 | 0.374 | 8.2 |
| 2 | 3 | [0.0073, 2.8844] | 1 | 6 | 0.416 | 1.3723 | 0.451 | 5.0 |
| 3 | 4 | [0.0073, 2.8844] | 2 | 7 | 0.517 | 1.1479 | 0.351 | 5.8 |
| 4 | 0 | [0.0083, 1.6282] | 3 | 8 | 0.476 | 1.2577 | 0.442 | 6.6 |
