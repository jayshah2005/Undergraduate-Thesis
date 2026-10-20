# 11 PMMFNet - out-of-fold results

Settings: lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, channels=32, bands=204 zero-filled to 208, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.398** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 44 | 64 | 81 | 37 | 226 |
| **RB only** | 98 | 304 | 51 | 64 | 517 |
| **LR only** | 32 | 11 | 72 | 24 | 139 |
| **both** | 18 | 13 | 51 | 34 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.539 | 0.496 |
| RB only | 517 | 0.800 | 0.534 |
| LR only | 139 | 0.729 | 0.516 |
| both | 116 | 0.665 | 0.512 |
| **mean** | | 0.683 | 0.514 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 229 | 136 | 218 | 415 | 0.642 | 0.709 | 0.558 |
| leafroll3 | 748 (254) | 269 | 225 | 73 | 181 | 0.629 | 0.650 | 0.487 |
| any_disease | 998 (772) | 44 | 182 | 148 | 624 | 0.501 | 0.539 | 0.496 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.245 | 0.503 | 0.551 | 0.538 | 0.472 | 0.451 | 0.416 | 0.401 |
| Cab Franc | 250 | 0.321 | 0.495 | 0.540 | 0.617 | n/a | n/a | 0.497 | 0.659 |
| Chardonnay | 250 | 0.300 | 0.525 | 0.580 | 0.602 | 0.565 | 0.506 | 0.568 | 0.532 |
| Vidal | 250 | 0.228 | 0.498 | 0.508 | 0.527 | 0.453 | 0.532 | 0.504 | 0.506 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [2, 1, 33, 23]; RB only -> [17, 1, 31, 33]; LR only -> [14, 0, 27, 13]; both -> [7, 1, 22, 23]
- **Cab Franc**: healthy -> [2, 6, 1, 0]; RB only -> [55, 178, 6, 1]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [32, 0, 47, 11]; RB only -> [12, 0, 14, 12]; LR only -> [16, 0, 45, 11]; both -> [7, 3, 29, 11]
- **Vidal**: healthy -> [8, 57, 0, 3]; RB only -> [14, 125, 0, 18]; LR only -> [2, 11, 0, 0]; both -> [4, 8, 0, 0]

## Early stopping

Validation = one training row per fold (not used for training); loss = leaf-level class-balanced cross-entropy. Full curves in the .json file.

| test fold | validation fold | best epoch | stopped after | best val loss | val bal. acc. at best |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 6 | 1.4072 | 0.313 |
| 1 | 2 | 1 | 6 | 1.2101 | 0.396 |
| 2 | 3 | 1 | 6 | 1.3467 | 0.382 |
| 3 | 4 | 2 | 7 | 1.1461 | 0.501 |
| 4 | 0 | 6 | 11 | 1.2524 | 0.489 |
