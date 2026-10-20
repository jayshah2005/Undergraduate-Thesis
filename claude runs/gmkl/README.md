# GMKL: graph-based multiple kernel learning

Hassanzadeh et al., *A novel graph-based multiple kernel learning framework for hyperspectral image classification*, Int. J. Remote Sens. 45:3075 (2024), applied to the 998 leaves with the 4 classes healthy / RB only / LR only / both.

| file | what it is |
|---|---|
| `gmkl.py` | the building blocks: DBSCAN superpixels, adjacent-superpixel kernel, attribute profiles, block-diagonal graph, kernel weights |
| `12_gmkl.py` | the pipeline: run at pixel level and at leaf level, each with two penalty graphs (pair / full). The paper-to-script mapping and deviations are in its docstring |
| `results/12_gmkl__<unit>__<penalty>_penalty_*` | out-of-fold predictions and metrics of the 4 runs |
| `logs/12.log` | the full-run log (`--quick` logs to `logs/quick_12.log`; new runs append to the file) |

```bash
cd "claude runs/gmkl"
../../.venv/bin/python -u 12_gmkl.py --quick     # ~1 min check (fold 0) -> results/quick/
../../.venv/bin/python -u 12_gmkl.py             # full run, ~35 min
```

It needs `../shared/cache/pmmf_pixel_index.npy`, which `../pmmfnet/11_pmmfnet.py` builds.

Code layout: `gmkl.py` functions are silent and return values; `12_gmkl.py` computes per fold (`run_leaf_fold`, `run_pixel_fold`, `run_folds`), then writes the reports (`write_reports`). Progress goes through the shared logger (`../shared/logs.py`) set up in `main()`, to the terminal and the log file. The kernel algebra runs in PyTorch on the GPU; the SVM is scikit-learn's LIBSVM `SVC` with a precomputed kernel (the paper's classifier).

## Result

| unit | 4-class balanced accuracy | mean OvR AUC pooled / within-variety | RB / LR within-variety AUC |
|---|---|---|---|
| pixel | 0.419 | 0.717 / 0.533 | 0.563 / 0.488 |
| leaf | 0.353 | 0.662 / 0.499 | 0.524 / 0.484 |

The two penalty graphs give identical results, because the learned kernel weights agree within 0.01.
