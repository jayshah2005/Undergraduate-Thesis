# PMMFNet custom run S2 + LR 10e-4 - S2 with lr = 3 x 10^-4 = 3e-5

Settings: folds=[0, 1, 2, 3, 4], seed=0, deterministic=False, data.scaler=train, model.channels=32, model.p_kernels=[7, 7, 4, 3, 3, 3], model.i_kernels=[7, 5, 4, 3, 3, 3], model.d_kernels=[5, 5, 5, 5, 4, 3], training.objective=patch, training.shuffle=True, training.lr=0.0003, training.weight_decay=0.0, training.batch_size=256, training.samples_per_class=100000, training.leaves_per_class=2, training.patches_per_leaf=32, early_stopping.metric=balanced_accuracy, early_stopping.patience=5, early_stopping.max_epochs=50, early_stopping.min_delta=0.0, evaluation.shuffle=True, evaluation.val_patches_per_leaf=256, evaluation.test_patches_per_leaf=0

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.415** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 51 | 75 | 62 | 38 | 226 |
| **RB only** | 53 | 366 | 43 | 55 | 517 |
| **LR only** | 29 | 17 | 59 | 34 | 139 |
| **both** | 14 | 22 | 45 | 35 | 116 |

Macro-F1: **0.400**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.347 | 0.226 | 0.273 | 226 |
| RB only | 0.762 | 0.708 | 0.734 | 517 |
| LR only | 0.282 | 0.424 | 0.339 | 139 |
| both | 0.216 | 0.302 | 0.252 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.624 | 0.526 |
| RB only | 517 | 0.805 | 0.529 |
| LR only | 139 | 0.748 | 0.528 |
| both | 116 | 0.689 | 0.491 |
| **mean** | | 0.716 | 0.519 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 201 | 164 | 155 | 478 | 0.653 | 0.728 | 0.562 |
| leafroll3 | 748 (254) | 303 | 191 | 81 | 173 | 0.647 | 0.683 | 0.513 |
| any_disease | 998 (772) | 51 | 175 | 96 | 676 | 0.551 | 0.624 | 0.526 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.319 | 0.540 | 0.563 | 0.596 | 0.554 | 0.529 | 0.586 | 0.532 |
| Cab Franc | 250 | 0.301 | 0.423 | 0.512 | 0.550 | n/a | n/a | 0.465 | 0.462 |
| Chardonnay | 250 | 0.258 | 0.501 | 0.560 | 0.578 | 0.505 | 0.475 | 0.525 | 0.522 |
| Vidal | 250 | 0.249 | 0.535 | 0.502 | 0.510 | 0.493 | 0.566 | 0.505 | 0.538 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [17, 6, 14, 22]; RB only -> [11, 9, 19, 43]; LR only -> [5, 5, 25, 19]; both -> [6, 5, 20, 22]
- **Cab Franc**: healthy -> [0, 8, 1, 0]; RB only -> [17, 217, 4, 2]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [27, 1, 46, 16]; RB only -> [10, 0, 18, 10]; LR only -> [23, 0, 34, 15]; both -> [7, 5, 25, 13]
- **Vidal**: healthy -> [7, 60, 1, 0]; RB only -> [15, 140, 2, 0]; LR only -> [1, 12, 0, 0]; both -> [1, 11, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 1 | 6 | 0.403 | 1.5245 | 0.379 | 5.3 |
| 1 | 2 | [0.0073, 2.8844] | 3 | 8 | 0.492 | 1.2708 | 0.411 | 6.8 |
| 2 | 3 | [0.0073, 2.8844] | 3 | 8 | 0.425 | 1.6356 | 0.505 | 6.7 |
| 3 | 4 | [0.0073, 2.8844] | 6 | 11 | 0.461 | 1.2686 | 0.306 | 9.2 |
| 4 | 0 | [0.0083, 1.6282] | 4 | 9 | 0.503 | 1.3863 | 0.401 | 7.6 |
