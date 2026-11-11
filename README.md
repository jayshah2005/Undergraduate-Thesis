# Grapevine hyperspectral disease classification

Classify red blotch and leafroll-3 on Specim IQ leaf cut-outs (998 leaves, 4 varieties, 204 bands, 397–1004 nm). Labels are binary per disease; a leaf can have both, either, or neither (healthy). Each vine is one leaf from `vine_hsi.h5`.

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

Cab Franc has essentially no leafroll positives, so leafroll training and scoring leave that variety out (it is still used for red blotch).

`data analysis.ipynb` builds these tables with the small loader `vine_data.py`; the analyses below use `helpers/data.py`.

---

## Layout

Analyses are grouped by what they ask:

| path | what it is |
|---|---|
| [`variety_classification/`](variety_classification/) | predict a leaf's variety: logistic regression and a whole-leaf CNN, each with all 204 bands or RGB only |
| [`spectral_analysis/`](spectral_analysis/) | what the spectra look like: PCA by variety and by disease class, plus the poster's normalisation figure |
| [`disease_classification/`](disease_classification/) | detect a disease within a variety: whole-leaf CNN with attention pooling vs the mean-spectrum baseline |

Shared code, in [`helpers/`](helpers/):

| file | what it is |
|---|---|
| `data.py` | leaf mean spectra (cached in `cache/`), whole-leaf cubes, leaf sizes, and which camera image each leaf came from |
| `pca_common.py` | SNV, PCA, the R² test, variety names and colours |
| `cnn_common.py` | masked batch norm, mask erosion, canvas padding, augmentation for the whole-leaf CNNs |
| `log.py` | terminal + log-file output |
| `report_tables.py` | the shared report format: every run gets `final_report.md` (balanced accuracy, Δ vs logistic regression, Fold 1 (R1) … Fold 5 (R5)) and `metrics.md` (every metric) |
| `diagram.py` | drawing pieces (cubes, boxes, arrows, legend) for the poster schematics: pipeline and the model architectures |
| `leaf_sources.csv` | leaf → camera image, matched by exact pixel values; read by `data.load_capture_ids` |

Evaluation: 5 folds by vineyard row, since neighbouring vines look alike. Fold k tests on row Rk of every variety, the next row is validation and the other 3 train. Disease models are scored by balanced accuracy, leafroll and red blotch separately.

Setup is in `requirements.txt`. Each folder's README says how to run it; scripts run with the repo's `.venv` from inside their folder. How code here is written: `.claude/skills/clean-analysis/`.
