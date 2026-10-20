# PMMFNet controlled experiment L2 - A2 with a leaf-level (multiple-instance) training loss

Settings: experiment=L2, scaler=train, train_shuffle=True, eval_shuffle=False, objective=leaf, lr=1e-05, batch=256, K per class=100000, max epochs=50, patience=5, val patches per leaf=256, channels=32, bands=204 zero-filled to 208, checkpoint=max val bal.acc, ties -> min val loss, folds=0,1,2,3,4

Every leaf is predicted by the model of the fold that held out its row. Decisions = the most probable of the 4 classes. 'Within-variety' AUC compares leaves only with leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).

## 4 classes

Balanced accuracy: **0.411** (chance = 0.25)

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | healthy | RB only | LR only | both | total |
|---|---|---|---|---|---|
| **healthy** | 59 | 61 | 74 | 32 | 226 |
| **RB only** | 91 | 315 | 45 | 66 | 517 |
| **LR only** | 37 | 13 | 55 | 34 | 139 |
| **both** | 22 | 16 | 34 | 44 | 116 |

Macro-F1: **0.393**

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.282 | 0.261 | 0.271 | 226 |
| RB only | 0.778 | 0.609 | 0.683 | 517 |
| LR only | 0.264 | 0.396 | 0.317 | 139 |
| both | 0.250 | 0.379 | 0.301 | 116 |

| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|
| healthy | 226 | 0.583 | 0.524 |
| RB only | 517 | 0.790 | 0.538 |
| LR only | 139 | 0.714 | 0.525 |
| both | 116 | 0.674 | 0.490 |
| **mean** | | 0.690 | 0.519 |

## Per disease (derived from the 4-class output)

| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |
|---|---|---|---|---|---|---|---|---|
| red_blotch | 998 (633) | 225 | 140 | 192 | 441 | 0.657 | 0.717 | 0.567 |
| leafroll3 | 748 (254) | 279 | 215 | 87 | 167 | 0.611 | 0.649 | 0.491 |
| any_disease | 998 (772) | 59 | 167 | 150 | 622 | 0.533 | 0.583 | 0.524 |

## Per variety

AUC here = within the variety, compared within folds. n/a = only one class present.

| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baco | 248 | 0.270 | 0.479 | 0.567 | 0.525 | 0.496 | 0.442 | 0.524 | 0.395 |
| Cab Franc | 250 | 0.372 | 0.469 | 0.559 | 0.659 | n/a | n/a | 0.563 | 0.710 |
| Chardonnay | 250 | 0.294 | 0.536 | 0.604 | 0.605 | 0.491 | 0.515 | 0.524 | 0.545 |
| Vidal | 250 | 0.247 | 0.509 | 0.528 | 0.560 | 0.473 | 0.555 | 0.524 | 0.576 |

4-class confusion matrices per variety (rows = true, columns = predicted: healthy / RB only / LR only / both):

- **Baco**: healthy -> [10, 0, 27, 22]; RB only -> [9, 1, 27, 45]; LR only -> [9, 1, 16, 28]; both -> [5, 1, 15, 32]
- **Cab Franc**: healthy -> [3, 6, 0, 0]; RB only -> [50, 188, 2, 0]; LR only -> [0, 0, 0, 0]; both -> [0, 1, 0, 0]
- **Chardonnay**: healthy -> [33, 0, 47, 10]; RB only -> [12, 1, 15, 10]; LR only -> [25, 2, 39, 6]; both -> [14, 5, 19, 12]
- **Vidal**: healthy -> [13, 55, 0, 0]; RB only -> [20, 125, 1, 11]; LR only -> [3, 10, 0, 0]; both -> [3, 9, 0, 0]

## Per fold

| test fold | val fold | scaler [min, max] | best epoch | stopped | best val bal. acc. | val loss at best | test bal. acc. | train min |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | [0.0077, 1.5846] | 1 | 6 | 0.368 | 1.3631 | 0.327 | 5.0 |
| 1 | 2 | [0.0073, 2.8844] | 1 | 6 | 0.443 | 1.1944 | 0.387 | 5.0 |
| 2 | 3 | [0.0073, 2.8844] | 5 | 10 | 0.467 | 1.3589 | 0.478 | 8.3 |
| 3 | 4 | [0.0073, 2.8844] | 3 | 8 | 0.460 | 1.2418 | 0.365 | 6.6 |
| 4 | 0 | [0.0083, 1.6282] | 14 | 19 | 0.438 | 1.3763 | 0.453 | 15.7 |
