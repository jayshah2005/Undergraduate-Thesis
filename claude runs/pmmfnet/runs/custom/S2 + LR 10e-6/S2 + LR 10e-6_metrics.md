# PMMFNet custom run S2 + LR 10e-6 - S2 with lr = 3 x 10e-6 = 3e-5 (3x the paper's 1e-5)

Settings: folds=[0, 1, 2, 3, 4], seed=0, deterministic=False, data.scaler=train, model.channels=32, model.p_kernels=[7, 7, 4, 3, 3, 3], model.i_kernels=[7, 5, 4, 3, 3, 3], model.d_kernels=[5, 5, 5, 5, 4, 3], training.objective=patch, training.shuffle=True, training.lr=3e-06, training.weight_decay=0.0, training.batch_size=256, training.samples_per_class=100000, training.leaves_per_class=2, training.patches_per_leaf=32, early_stopping.metric=balanced_accuracy, early_stopping.patience=5, early_stopping.max_epochs=50, early_stopping.min_delta=0.0, evaluation.shuffle=True, evaluation.val_patches_per_leaf=256, evaluation.test_patches_per_leaf=0

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.402** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 45 | 65 | 72 | 44 | 226 |
| **RB only** | 86 | 324 | 47 | 60 | 517 |
| **LR only** | 32 | 11 | 57 | 39 | 139 |
| **both** | 25 | 14 | 34 | 43 | 116 |

Macro-F1: **0.381**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.239 | 0.199 | 0.217 | 226 |
| RB only | 0.783 | 0.627 | 0.696 | 517 |
| LR only | 0.271 | 0.410 | 0.327 | 139 |
| both | 0.231 | 0.371 | 0.285 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.561 | 0.496 |
| RB only | 517 | 0.797 | 0.533 |
| LR only | 139 | 0.716 | 0.517 |
| both | 116 | 0.683 | 0.548 |
| **mean** | | 0.689 | 0.524 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 206 | 159 | 192 | 441 | 0.631 | 0.710 | 0.566 |
| leafroll3 | 748 (254) | 280 | 214 | 81 | 173 | 0.624 | 0.654 | 0.501 |
| any_disease | 998 (772) | 45 | 181 | 143 | 629 | 0.507 | 0.561 | 0.496 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.245 | 0.503 | 0.495 | 0.524 | 0.487 | 0.477 | 0.451 | 0.369 |
| Cab Franc | 250 | 0.379 | 0.469 | 0.573 | 0.582 | n/a | n/a | 0.588 | 0.615 |
| Chardonnay | 250 | 0.283 | 0.523 | 0.585 | 0.629 | 0.501 | 0.509 | 0.514 | 0.537 |
| Vidal | 250 | 0.242 | 0.519 | 0.503 | 0.548 | 0.502 | 0.543 | 0.508 | 0.536 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [8, 0, 17, 34]; RB only -> [20, 0, 22, 40]; LR only -> [13, 0, 17, 24]; both -> [11, 0, 14, 28]
- **Cab Franc**: healthy -> [3, 6, 0, 0]; RB only -> [38, 193, 7, 2]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [25, 0, 55, 10]; RB only -> [10, 0, 18, 10]; LR only -> [18, 0, 40, 14]; both -> [12, 3, 20, 15]
- **Vidal**: healthy -> [9, 59, 0, 0]; RB only -> [18, 131, 0, 8]; LR only -> [1, 11, 0, 1]; both -> [2, 10, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 10 | 15 | 0.357 | 1.4820 | 0.298 | 12.5 |
| 1 | 2 | [0.0073, 2.8844] | 2 | 7 | 0.462 | 1.1841 | 0.396 | 5.9 |
| 2 | 3 | [0.0073, 2.8844] | 1 | 6 | 0.431 | 1.3099 | 0.498 | 5.2 |
| 3 | 4 | [0.0073, 2.8844] | 2 | 7 | 0.510 | 1.1798 | 0.377 | 5.9 |
| 4 | 0 | [0.0083, 1.6282] | 9 | 14 | 0.455 | 1.2568 | 0.402 | 11.9 |
