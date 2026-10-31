# clean/

Each vine is one leaf from `vine_hsi.h5`. Analyses are grouped by what they ask.

| path | what it is |
|---|---|
| [`variety_classification/`](variety_classification/) | predict a leaf's variety: logistic regression and a whole-leaf CNN, each with all 204 bands or RGB only |
| [`spectral_analysis/`](spectral_analysis/) | what the spectra look like: PCA by variety and by disease class, plus the poster's normalisation figure |
| [`disease_classification/`](disease_classification/) | detect a disease within a variety: whole-leaf CNN with attention pooling vs the mean-spectrum baseline |

Shared code, used by the folders above:

| file | what it is |
|---|---|
| `data.py` | leaf mean spectra (cached in `cache/`), whole-leaf cubes, leaf sizes, and which camera image each leaf came from |
| `cnn_common.py` | masked batch norm, mask erosion, canvas padding, augmentation for the whole-leaf CNNs |
| `log.py` | terminal + log-file output |
| `leaf_sources.csv` | leaf → camera image, matched by exact pixel values; read by `data.load_capture_ids` |

How code here must be written: `../.claude/skills/clean-analysis/`. Each folder's README says how to run it.
