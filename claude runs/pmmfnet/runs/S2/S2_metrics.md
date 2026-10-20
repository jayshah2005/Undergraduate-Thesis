# PMMFNet controlled experiment S2 - A2 + fixed spatial shuffle of validation and test patches

Settings: experiment=S2, scaler=train, train_shuffle=True, eval_shuffle=True, objective=patch, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.403** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 36 | 56 | 103 | 31 | 226 |
| **RB only** | 83 | 323 | 48 | 63 | 517 |
| **LR only** | 15 | 14 | 73 | 37 | 139 |
| **both** | 13 | 17 | 51 | 35 | 116 |

Macro-F1: **0.373**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.245 | 0.159 | 0.193 | 226 |
| RB only | 0.788 | 0.625 | 0.697 | 517 |
| LR only | 0.265 | 0.525 | 0.353 | 139 |
| both | 0.211 | 0.302 | 0.248 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.553 | 0.502 |
| RB only | 517 | 0.812 | 0.538 |
| LR only | 139 | 0.736 | 0.526 |
| both | 116 | 0.670 | 0.513 |
| **mean** | | 0.693 | 0.519 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 227 | 138 | 195 | 438 | 0.657 | 0.729 | 0.577 |
| leafroll3 | 748 (254) | 256 | 238 | 58 | 196 | 0.645 | 0.653 | 0.488 |
| any_disease | 998 (772) | 36 | 190 | 111 | 661 | 0.508 | 0.553 | 0.502 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.257 | 0.522 | 0.584 | 0.584 | 0.496 | 0.505 | 0.471 | 0.473 |
| Cab Franc | 250 | 0.346 | 0.480 | 0.530 | 0.566 | n/a | n/a | 0.534 | 0.541 |
| Chardonnay | 250 | 0.275 | 0.524 | 0.562 | 0.599 | 0.533 | 0.468 | 0.510 | 0.484 |
| Vidal | 250 | 0.250 | 0.498 | 0.547 | 0.547 | 0.480 | 0.498 | 0.546 | 0.539 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [6, 2, 35, 16]; RB only -> [13, 7, 25, 37]; LR only -> [10, 2, 25, 17]; both -> [7, 3, 23, 20]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [37, 196, 2, 5]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [12, 0, 63, 15]; RB only -> [9, 0, 17, 12]; LR only -> [4, 0, 48, 20]; both -> [5, 3, 27, 15]
- **Vidal**: healthy -> [16, 47, 5, 0]; RB only -> [24, 120, 4, 9]; LR only -> [1, 12, 0, 0]; both -> [1, 10, 1, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 5 | 10 | 0.340 | 1.5104 | 0.305 | 8.3 |
| 1 | 2 | [0.0073, 2.8844] | 5 | 10 | 0.428 | 1.2600 | 0.383 | 8.3 |
| 2 | 3 | [0.0073, 2.8844] | 6 | 11 | 0.400 | 1.4436 | 0.494 | 9.1 |
| 3 | 4 | [0.0073, 2.8844] | 2 | 7 | 0.500 | 1.1501 | 0.353 | 5.8 |
| 4 | 0 | [0.0083, 1.6282] | 9 | 14 | 0.498 | 1.2970 | 0.413 | 11.6 |
