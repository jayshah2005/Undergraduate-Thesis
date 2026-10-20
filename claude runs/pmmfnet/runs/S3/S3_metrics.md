# PMMFNet controlled experiment S3 - shuffle in training only (= A2 settings, rerun)

Settings: experiment=S3, scaler=train, train_shuffle=True, eval_shuffle=False, objective=patch, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.406** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 30 | 70 | 98 | 28 | 226 |
| **RB only** | 87 | 334 | 49 | 47 | 517 |
| **LR only** | 21 | 14 | 77 | 27 | 139 |
| **both** | 11 | 20 | 51 | 34 | 116 |

Macro-F1: **0.375**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.201 | 0.133 | 0.160 | 226 |
| RB only | 0.763 | 0.646 | 0.699 | 517 |
| LR only | 0.280 | 0.554 | 0.372 | 139 |
| both | 0.250 | 0.293 | 0.270 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.557 | 0.486 |
| RB only | 517 | 0.804 | 0.518 |
| LR only | 139 | 0.733 | 0.517 |
| both | 116 | 0.669 | 0.514 |
| **mean** | | 0.691 | 0.509 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 226 | 139 | 198 | 435 | 0.653 | 0.727 | 0.560 |
| leafroll3 | 748 (254) | 275 | 219 | 65 | 189 | 0.650 | 0.655 | 0.484 |
| any_disease | 998 (772) | 30 | 196 | 119 | 653 | 0.489 | 0.557 | 0.486 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.264 | 0.505 | 0.568 | 0.544 | 0.496 | 0.482 | 0.478 | 0.431 |
| Cab Franc | 250 | 0.306 | 0.477 | 0.520 | 0.580 | n/a | n/a | 0.464 | 0.538 |
| Chardonnay | 250 | 0.288 | 0.525 | 0.574 | 0.609 | 0.536 | 0.478 | 0.512 | 0.497 |
| Vidal | 250 | 0.237 | 0.481 | 0.496 | 0.525 | 0.484 | 0.504 | 0.497 | 0.510 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [9, 1, 32, 17]; RB only -> [18, 5, 31, 28]; LR only -> [14, 2, 24, 14]; both -> [5, 5, 22, 21]
- **Cab Franc**: healthy -> [1, 7, 1, 0]; RB only -> [44, 194, 0, 2]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [14, 0, 65, 11]; RB only -> [10, 0, 18, 10]; LR only -> [6, 0, 53, 13]; both -> [5, 3, 29, 13]
- **Vidal**: healthy -> [6, 62, 0, 0]; RB only -> [15, 135, 0, 7]; LR only -> [1, 12, 0, 0]; both -> [1, 11, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 5 | 10 | 0.344 | 1.5091 | 0.325 | 8.3 |
| 1 | 2 | [0.0073, 2.8844] | 1 | 6 | 0.416 | 1.2088 | 0.374 | 5.0 |
| 2 | 3 | [0.0073, 2.8844] | 3 | 8 | 0.399 | 1.3492 | 0.489 | 6.6 |
| 3 | 4 | [0.0073, 2.8844] | 2 | 7 | 0.507 | 1.1498 | 0.353 | 5.8 |
| 4 | 0 | [0.0083, 1.6282] | 8 | 13 | 0.497 | 1.3291 | 0.412 | 10.8 |
