# PMMFNet controlled experiment A1 - original configuration, balanced-accuracy checkpoint selection

Settings: experiment=A1, scaler=train+val, train_shuffle=True, eval_shuffle=False, objective=patch, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.413** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 39 | 64 | 94 | 29 | 226 |
| **RB only** | 70 | 337 | 54 | 56 | 517 |
| **LR only** | 22 | 12 | 73 | 32 | 139 |
| **both** | 15 | 19 | 47 | 35 | 116 |

Macro-F1: **0.385**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.267 | 0.173 | 0.210 | 226 |
| RB only | 0.780 | 0.652 | 0.710 | 517 |
| LR only | 0.272 | 0.525 | 0.359 | 139 |
| both | 0.230 | 0.302 | 0.261 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.569 | 0.491 |
| RB only | 517 | 0.805 | 0.523 |
| LR only | 139 | 0.744 | 0.536 |
| both | 116 | 0.672 | 0.498 |
| **mean** | | 0.698 | 0.512 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 228 | 137 | 186 | 447 | 0.665 | 0.726 | 0.558 |
| leafroll3 | 748 (254) | 269 | 225 | 67 | 187 | 0.640 | 0.666 | 0.482 |
| any_disease | 998 (772) | 39 | 187 | 107 | 665 | 0.517 | 0.569 | 0.491 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.276 | 0.512 | 0.595 | 0.536 | 0.471 | 0.484 | 0.490 | 0.460 |
| Cab Franc | 250 | 0.356 | 0.519 | 0.592 | 0.580 | n/a | n/a | 0.549 | 0.606 |
| Chardonnay | 250 | 0.275 | 0.517 | 0.564 | 0.610 | 0.540 | 0.462 | 0.526 | 0.476 |
| Vidal | 250 | 0.245 | 0.496 | 0.515 | 0.527 | 0.480 | 0.525 | 0.521 | 0.513 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [10, 0, 31, 18]; RB only -> [13, 3, 30, 36]; LR only -> [14, 0, 26, 14]; both -> [9, 3, 19, 22]
- **Cab Franc**: healthy -> [2, 6, 1, 0]; RB only -> [30, 203, 6, 1]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [17, 0, 62, 11]; RB only -> [10, 0, 18, 10]; LR only -> [7, 0, 47, 18]; both -> [5, 4, 28, 13]
- **Vidal**: healthy -> [10, 58, 0, 0]; RB only -> [17, 131, 0, 9]; LR only -> [1, 12, 0, 0]; both -> [1, 11, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.6282] | 3 | 8 | 0.362 | 1.4288 | 0.345 | 6.6 |
| 1 | 2 | [0.0073, 2.8844] | 1 | 6 | 0.416 | 1.2088 | 0.374 | 5.0 |
| 2 | 3 | [0.0073, 2.8844] | 6 | 11 | 0.416 | 1.4504 | 0.465 | 9.1 |
| 3 | 4 | [0.0073, 2.8844] | 2 | 7 | 0.515 | 1.1447 | 0.355 | 5.8 |
| 4 | 0 | [0.0073, 2.8844] | 8 | 13 | 0.485 | 1.3251 | 0.441 | 10.7 |
