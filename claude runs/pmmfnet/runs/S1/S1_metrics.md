# PMMFNet controlled experiment S1 - A2 without spatial shuffle anywhere

Settings: experiment=S1, scaler=train, train_shuffle=False, eval_shuffle=False, objective=patch, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.437** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 54 | 61 | 80 | 31 | 226 |
| **RB only** | 77 | 348 | 36 | 56 | 517 |
| **LR only** | 32 | 12 | 66 | 29 | 139 |
| **both** | 19 | 17 | 38 | 42 | 116 |

Macro-F1: **0.417**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.297 | 0.239 | 0.265 | 226 |
| RB only | 0.795 | 0.673 | 0.729 | 517 |
| LR only | 0.300 | 0.475 | 0.368 | 139 |
| both | 0.266 | 0.362 | 0.307 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.602 | 0.516 |
| RB only | 517 | 0.812 | 0.545 |
| LR only | 139 | 0.749 | 0.533 |
| both | 116 | 0.696 | 0.524 |
| **mean** | | 0.715 | 0.530 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 232 | 133 | 170 | 463 | 0.684 | 0.747 | 0.580 |
| leafroll3 | 748 (254) | 300 | 194 | 79 | 175 | 0.648 | 0.680 | 0.509 |
| any_disease | 998 (772) | 54 | 172 | 128 | 644 | 0.537 | 0.602 | 0.516 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.293 | 0.523 | 0.571 | 0.584 | 0.506 | 0.506 | 0.501 | 0.451 |
| Cab Franc | 250 | 0.362 | 0.449 | 0.555 | 0.566 | n/a | n/a | 0.561 | 0.571 |
| Chardonnay | 250 | 0.302 | 0.543 | 0.603 | 0.614 | 0.550 | 0.513 | 0.519 | 0.534 |
| Vidal | 250 | 0.269 | 0.506 | 0.545 | 0.542 | 0.489 | 0.504 | 0.549 | 0.542 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [17, 0, 19, 23]; RB only -> [25, 2, 18, 37]; LR only -> [16, 0, 21, 17]; both -> [13, 3, 12, 25]
- **Cab Franc**: healthy -> [2, 7, 0, 0]; RB only -> [24, 207, 3, 6]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [22, 0, 60, 8]; RB only -> [14, 0, 15, 9]; LR only -> [15, 0, 45, 12]; both -> [4, 3, 26, 17]
- **Vidal**: healthy -> [13, 54, 1, 0]; RB only -> [14, 139, 0, 4]; LR only -> [1, 12, 0, 0]; both -> [2, 10, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 6 | 11 | 0.362 | 1.5129 | 0.345 | 9.1 |
| 1 | 2 | [0.0073, 2.8844] | 6 | 11 | 0.451 | 1.2422 | 0.447 | 9.0 |
| 2 | 3 | [0.0073, 2.8844] | 4 | 9 | 0.390 | 1.4036 | 0.521 | 7.4 |
| 3 | 4 | [0.0073, 2.8844] | 3 | 8 | 0.526 | 1.1411 | 0.349 | 6.6 |
| 4 | 0 | [0.0083, 1.6282] | 4 | 9 | 0.487 | 1.2884 | 0.415 | 7.4 |
