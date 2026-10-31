# claude runs: index

Everything here works on the 998 leaves of `../vine_hsi.h5`. Every folder has its own README (how to run it) and a report of what it found. Run each script from inside its folder with `../../.venv/bin/python -u <script>`.

| folder | what it is | headline |
|---|---|---|
| `shared/` | data cache (`cache/`) and the code every pipeline imports: folds, labels, metrics, leaf spectra | — |
| `pmmfnet/` | PMMFNet paper: the original run, the controlled experiments of `../plan.md`, and `run_experiment.py` + `configs/` for your own runs | best config S1, 4-class bal. acc. 0.44; within-variety AUC ≈ 0.53 |
| `gmkl/` | GMKL paper (graph-based multiple kernel learning), pixel and leaf level | 4-class bal. acc. 0.42 (pixel); within-variety AUC ≈ 0.52 |
| `spectra_exploration/` | Do the leaf spectra separate by variety or by disease? (PCA, variance split, effect sizes) | variety explains ~50% of the variance, disease within variety ≤ 1% |
| `disease_differences/` | healthy vs red blotch vs leafroll inside each of the 4 varieties | only Chardonnay healthy vs RB differs significantly (shape, red edge ~703 nm) |
| `red_blotch_spectra/` | red blotch within Chardonnay from leaf spectra: 5 feature sets (mean spectrum, derivative, vegetation indices, index spread across the leaf) × logistic / PLS-DA / gradient boosting × 3 fold designs, with a pre-declared permutation test | no pipeline beats the mean-spectrum baseline; within-row AUC ≈ 0.54–0.64; signal in the red edge and green/red ratio |
| `variety_logistic_regression/` | predicting the variety from the leaf spectrum (204 bands) | balanced accuracy 0.94 |
| `variety_logistic_regression_rgb/` | the same model on three pseudo-RGB bands (640 / 550 / 460 nm) | raw 0.86; SNV 0.50 |
| `cache/transfer_learning_study/` | archive of the first 47-run study (classical models, pixel nets, CNNs, foundation models) | all tied; within-variety AUC 0.53–0.59 |

Evaluation everywhere: 5 folds = vineyard rows R1–R5 held out whole (`red_blotch_spectra/` also uses photo- and block-based folds). Within-variety AUC (0.5 = chance) is the honest disease measure; pooled scores are inflated by recognising the variety.
