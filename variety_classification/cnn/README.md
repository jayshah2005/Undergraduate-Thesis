# Variety prediction with a whole-leaf CNN

Does a small CNN that sees the whole leaf do better than logistic regression on its mean spectrum, and do 204 bands beat RGB?

| path | what it is |
|---|---|
| `model.py` | `VarietyCNN`: a backbone (below) → masked mean pooling → 4 logits; `SECOND_BACKBONE` picks the second one in reports |
| `run_experiments.py` | padding check → train every (seed, fold) of one configuration; skips fold files already on disk |
| `report.py` | metrics, paired differences and figures for both inputs, with logistic regression as the comparison |
| `plot_architecture.py` | poster schematic of the network → `figures/architecture.png` |

| setting | choices | meaning |
|---|---|---|
| `--bands` | `all` (default), `rgb` | all 204 bands, or the 3 bands nearest 640 / 550 / 460 nm. Only the input of the first layer changes |
| `--backbone` | `small_cnn` (default), see below | the layers before the pooling; another adds `_<backbone>` to the run name |

## Set-up

- **Input:** each pixel gets SNV over its own bands, which removes brightness. Each leaf is centred on one 212 × 212 canvas (the largest leaf side, rounded up to a multiple of 4). Padding is 0 and a mask marks leaf pixels. Normalisation and pooling use leaf positions only, so padding can't change a prediction; each run checks this at start-up.
- **Splits:** fold k tests on row Rk of every variety (each variety has its own rows R1–R5), the next row is validation (early stopping), and the other 3 train. 5 folds × 3 seeds = 15 trainings per configuration.
- **Training:** batch 8, AdamW (lr 1e-3, weight decay 1e-2), dropout 0.5, cross-entropy, up to 80 epochs with patience 15. Augmentation is 90° rotations and flips, with no brightness or band jitter. `--no-augmentation` (on both `run_experiments.py` and `report.py`) trains without it, into `results/<bands>_no_augmentation/`, and the report files get the same suffix.
- **Comparison:** logistic regression with the same split: it fits on the same 3 rows and chooses C on the validation row, so neither model has more data.

## Backbones

The layers before the pooling come from `helpers/backbones.py` and are picked with `--backbone`. Every backbone shrinks the map exactly 4× and uses masked batch norm, so the padding check still passes, and all train with the same settings. Reports show the default and `SECOND_BACKBONE` from `model.py`, the best of the others on the validation rows.

| `--backbone` | from | changed for whole leaves |
|---|---|---|
| `small_cnn` (default) | this repo's own design | — |
| `small_resnet` | residual blocks, He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR | 4 basic blocks (32, 64, 64, 64 channels) instead of ResNet-18's 8 |
| `conv3d` | HybridSN, Roy et al. 2020, IEEE GRSL 17(2):277–281; a 3D CNN on Specim IQ grapevine leaves: Nguyen et al. 2021, Sensors 21(3):742 | a learned 1×1 mix to 24 bands replaces PCA; map halved after the first 3D layer (HybridSN uses 25×25 patches) |
| `spectral_1d` | 1D spectral CNN, Hu et al. 2015, "Deep Convolutional Neural Networks for Hyperspectral Image Classification", J. Sensors 2015:258619 | applied at every leaf position after a 2×2 masked average; no spatial layers |
| `resnet18_imagenet` | ImageNet ResNet-18 (He et al. 2016) behind a linear band adaptor, Pérez & Maji 2021, arXiv:2108.01555 | cut after layer2, no max pool, batch norms masked |

```bash
cd variety_classification/cnn
../../.venv/bin/python -u run_experiments.py --bands rgb --smoke   # ~40 s: 1 fold, 3 epochs, results/smoke/
../../.venv/bin/python -u run_experiments.py --bands all           # resumable
../../.venv/bin/python -u run_experiments.py --bands rgb
../../.venv/bin/python -u run_experiments.py --bands all --backbone small_resnet   # another backbone, results/all_small_resnet/
../../.venv/bin/python -u report.py                                # results/<bands>/{final_report,metrics}.md + figures/; needs both runs
../../.venv/bin/python -u plot_architecture.py                     # figures/architecture.png
```

| output | what it is |
|---|---|
| `results/<bands>/seed<s>/fold<k>.npz` | test and validation probabilities and loss curves of one training |
| `results/<bands>/config.json` | settings; a folder only resumes runs made with the same ones |
| `results/<bands>_<backbone>/` | another backbone's runs, with its own `final_report.md` and `metrics.md`; its rows also sit in `results/<bands>/`'s reports |
| `results/<bands>/final_report.md` | balanced accuracy of every model with 95% CI, Δ vs logistic regression, per-fold scores; recall per variety, single seeds, 204 bands − RGB |
| `results/<bands>/metrics.md` | every metric of every model (accuracy, macro F1, per-variety sensitivity, specificity, precision, F1, confusion counts) |
| `figures/scores.png`, `figures/confusion_<bands>.png` | all models compared; the CNN's confusion matrices |
| `logs/<bands>.log` | progress of the last training: one line per seed |

Results and what they mean: [`../REPORT.md`](../REPORT.md).
