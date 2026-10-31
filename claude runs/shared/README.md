# shared: data cache and common code

Imported by every pipeline in `claude runs/` (each script adds this folder to `sys.path`).

| file | what it does |
|---|---|
| `common.py` | paths, labels, the 5 folds (fold k = row R(k+1) of every variety; whole rows held out), data loaders, AUC helpers incl. within-variety AUC |
| `four_class.py` | the 4-class formulation (healthy / RB only / LR only / both) and the report the paper pipelines write: confusion matrix, balanced accuracy, macro-F1, ROC AUC (pooled and within-variety), per disease, per variety |
| `spectra.py` | one mean spectrum per leaf (cached in `cache/leaf_mean_spectra.npz`) and SNV normalisation; used by the three analysis folders |
| `logs.py` | `get_logger(name, log_file)`: the logger every script sets up in `main()` (terminal + log file); computation code never prints. `save_results`, `save_oof`, `print_fold_summary` and `train_model` take an optional `log=` to send their lines to it |
| `torch_common.py` | device and seeding helpers (plus the training loop of the archived scripts) |
| `00_build_cache.py` | reads `../../vine_hsi.h5` once and writes `cache/`; log in `logs/00.log` |

## cache/

| file | contents | built by |
|---|---|---|
| `meta.parquet` | one row per leaf: path, variety, row, vine, labels, fold, pixel offsets | `00_build_cache.py` |
| `pixels.npy` | every leaf pixel, 204 bands (7.7M × 204, float32) | `00_build_cache.py` |
| `wavelengths.npy` | band centres (nm) | `00_build_cache.py` |
| `images.npy`, `masks.npy` | whole leaves as 128 × 128 images (6.3 GB); used only by the archived scripts 06/07 | `00_build_cache.py` |
| `leaf_stats.npy` | per-leaf mean / SD / p10 / p90 per band; used only by archived scripts 02/10 | archived script 02 |
| `pmmf_pixel_index.npy` | native-resolution pixel positions | `../pmmfnet/11_pmmfnet.py` |
| `gmkl_*` | superpixels and leaf kernels | `../gmkl/12_gmkl.py` |
| `leaf_mean_spectra.npz` | mean spectrum + glint share per leaf | `spectra.py` |

**The cache is not on disk: it was deleted on 2026-10-30 to save 14 GB.** Scripts under `claude runs/` need it rebuilt before they can run.

Rebuild the base cache (a few minutes): `cd "claude runs/shared" && ../../.venv/bin/python -u 00_build_cache.py`.
