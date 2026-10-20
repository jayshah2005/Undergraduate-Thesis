# 11 PMMFNet - out-of-fold results

Settings: lr=1e-05, batch=256, K per class=2000, max epochs=3, patience=1, channels=32, bands=204 zero-filled to 208, folds=0

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.363** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 3 | 16 | 4 | 6 | 29 |
| **RB only** | 4 | 84 | 2 | 18 | 108 |
| **LR only** | 3 | 5 | 4 | 16 | 28 |
| **both** | 5 | 10 | 5 | 15 | 35 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 29 | 0.571 | 0.541 |
| RB only | 108 | 0.740 | 0.574 |
| LR only | 28 | 0.686 | 0.415 |
| both | 35 | 0.642 | 0.424 |
| **mean** | | 0.660 | 0.488 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 200 (143) | 14 | 43 | 16 | 127 | 0.567 | 0.629 | 0.512 |
| leafroll3 | 150 (63) | 57 | 30 | 23 | 40 | 0.645 | 0.671 | 0.468 |
| any_disease | 200 (171) | 3 | 26 | 12 | 159 | 0.517 | 0.571 | 0.541 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 50 | 0.250 | 0.408 | 0.500 | 0.441 | 0.500 | 0.484 | 0.500 | 0.177 |
| Cab Franc | 50 | 0.500 | 0.797 | 0.500 | 0.333 | n/a | n/a | 0.500 | 0.854 |
| Chardonnay | 50 | 0.252 | 0.492 | 0.484 | 0.580 | 0.530 | 0.472 | 0.440 | 0.478 |
| Vidal | 50 | 0.250 | 0.482 | 0.500 | 0.545 | 0.488 | 0.425 | 0.500 | 0.672 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [0, 0, 0, 3]; RB only -> [0, 0, 0, 18]; LR only -> [0, 0, 0, 15]; both -> [0, 0, 0, 14]
- **Cab Franc**: healthy -> [0, 2, 0, 0]; RB only -> [0, 48, 0, 0]; LR only -> [0, 0, 0, 0]; both -> [0, 0, 0, 0]
- **Chardonnay**: healthy -> [3, 5, 4, 2]; RB only -> [4, 3, 2, 0]; LR only -> [3, 2, 4, 1]; both -> [5, 6, 5, 1]
- **Vidal**: healthy -> [0, 9, 0, 1]; RB only -> [0, 33, 0, 0]; LR only -> [0, 3, 0, 0]; both -> [0, 4, 0, 0]

## Early stopping

Validation = one training row per fold (not used for training); loss = leaf-level class-balanced cross-entropy. Full curves in the .json file.

| test fold | validation fold | best epoch | stopped after | best val loss | val bal. acc. at best |
|---|---|---|---|---|---|
| 0 | 1 | 3 | 3 | 1.2837 | 0.397 |
