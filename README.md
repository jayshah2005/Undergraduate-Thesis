# Grapevine hyperspectral disease classification

Classify red blotch and leafroll-3 on Specim IQ leaf cut-outs (998 leaves, 4 varieties, 204 bands, 397–1004 nm). Labels are binary per disease; a leaf can have both, either, or neither (healthy).

This repo holds the raw HDF5, shared loaders, exploratory analysis, and paper re-implementations under `claude runs/` (index: `claude runs/README.md`). An earlier 47-run transfer-learning study is archived in `claude runs/cache/transfer_learning_study/` (weak within-variety signal; variety alone explains much of the pooled AUC).

---

## Dataset

| | |
|---|---|
| File | `vine_hsi.h5` (~4 GB) |
| Sensor | Specim IQ |
| Cubes | `(lines, bands, samples)` as stored in ENVI BIL |
| Bands | 204, 397–1004 nm (Specim IQ factory calibration, uneven ~3 nm steps) |
| Unit | one leaf cut-out (“vine”) |
| Labels | `red_blotch`, `leafroll3` (attrs on each vine group) |
| Varieties | Baco, Cab Franc, Chardonnay, Vidal |

### Label distribution (overlapping positives; Healthy = neither)

| Variety | n | Red blotch | Leafroll | Healthy |
|---|---:|---|---|---|
| Baco | 248 | 135 (54.4%) | 107 (43.1%) | 59 (23.8%) |
| Cab Franc | 250 | 241 (96.4%) | 1 (0.4%) | 9 (3.6%) |
| Chardonnay | 250 | 88 (35.2%) | 122 (48.8%) | 90 (36%) |
| Vidal | 250 | 169 (67.6%) | 25 (10%) | 68 (27.2%) |
| **All** | **998** | **633** | **255** | **226** |

### Mutually exclusive status (sums to 100% per variety)

| Variety | Red blotch only | Leafroll only | Both | Healthy |
|---|---|---|---|---|
| Baco | 82 (33.1%) | 54 (21.8%) | 53 (21.4%) | 59 (23.8%) |
| Cab Franc | 240 (96%) | 0 (0%) | 1 (0.4%) | 9 (3.6%) |
| Chardonnay | 38 (15.2%) | 72 (28.8%) | 50 (20%) | 90 (36%) |
| Vidal | 157 (62.8%) | 13 (5.2%) | 12 (4.8%) | 68 (27.2%) |
| **All** | **517** | **139** | **116** | **226** |

Cab Franc has essentially no leafroll positives, so leafroll training/scoring excludes that variety (still used for red blotch).

Shared metadata loader (notebooks): `vine_data.py` — `open_h5`, `load_wavelengths`, `index_vines`.

---

## Layout

```
Classification/
├── vine_hsi.h5              # raw cubes + labels
├── vine_data.py             # thin H5 index/loaders for notebooks
├── data analysis.ipynb      # EDA / label tables
├── plan.md                  # plan of the controlled PMMFNet experiments
├── requirements.txt
├── Knowledge Base.txt       # short notes / paper list
├── research_notes/          # longer write-ups
├── clean/                   # variety classification, spectral analysis, disease classification
└── claude runs/             # all code, results and reports (index: claude runs/README.md)
    ├── shared/                      # data cache + code every pipeline uses (folds, labels, metrics)
    ├── pmmfnet/                     # PMMFNet paper: original run, controlled experiments, your own runs
    ├── gmkl/                        # GMKL paper
    ├── spectra_exploration/         # do the spectra separate by variety or by disease?
    ├── disease_differences/         # healthy vs red blotch vs leafroll inside each variety
    ├── variety_logistic_regression/ # predicting the variety from the spectrum
    └── cache/transfer_learning_study/   # archived first 47-run study
```

Eval protocol (all scripts): 5 folds = whole vineyard **rows** held out together (neighbours look alike). Prefer **within-variety** ROC AUC (0.5 = chance; variety recognition does not help) over pooled AUC.

Each folder has a README with how to run it and a REPORT (or results section) with what it found. The setup is in `requirements.txt`; scripts run with `../../.venv/bin/python` from inside their folder.
