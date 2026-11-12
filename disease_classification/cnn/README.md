# Whole-leaf CNN, pooled varieties

Does one small CNN trained on the leaves of several varieties find disease better than the mean-spectrum baseline
(`../logistic_regression/`)? Leaves, labels, folds, baseline and scoring all come from that folder. The network and
training code here are also used by `../within_variety_disease_cnn/`.

| file | what it is |
|---|---|
| `model.py` | `LeafCNN`: a backbone (below) → mean or attention pooling over leaf pixels → one output; `SECOND_BACKBONE` picks the second one in reports |
| `train.py` | SNV per pixel, rotations and flips, training one fold (early stopping on the validation row); `CANVAS = 212` |
| `run_experiments.py` | trains every task × pooling × seed × fold of one pool; skips folds already saved; writes `logs/<run>.log` |
| `report.py` | `final_report.md`, `metrics.md` and figures for one run |
| `plot_architecture.py` | poster schematic of the network → `figures/architecture.png` |
| `results/<run>/` | `final_report.md` (balanced accuracy; the default run's report also has the second backbone's rows), `metrics.md` (every metric), `config.json`, saved test and validation predictions per fold |
| `figures/<run>/` | `balanced_accuracy.png`, `attention_maps.png` |

**Pools.** `all` (998 leaves) or `baco_chardonnay_vidal` (748). A variety with fewer than 5 positive or negative
leaves sits out a task, so Cab Franc drops out of leafroll and both pools train leafroll on the same 748 leaves.

**Models.** Each pixel's spectrum is SNV-normalised, as in the variety CNN, on a 212 × 212 canvas. Two models per task:
mean pooling or gated attention pooling (Ilse et al. 2018, "Attention-based Deep Multiple Instance
Learning", ICML) over leaf pixels. Each is trained with 3 seeds; the score uses their average probability.

**Score.** Balanced accuracy at P ≥ 0.5, as for the baseline, with a 95% interval, the paired difference (Δ) to the
baseline, and the score on each variety's leaves.

**Run names.** `<pool>[_rgb][_no_augmentation][_<backbone>]`, e.g. `all`, `baco_chardonnay_vidal_rgb`. Give `report.py` the same
flags as `run_experiments.py`, without `--backbone`: the default run's report also writes the second backbone's own
`final_report.md` and `metrics.md` in its `_<backbone>` folder.

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
cd disease_classification/cnn
../../.venv/bin/python -u run_experiments.py                                     # all 998 leaves, leafroll + red blotch
../../.venv/bin/python -u report.py
../../.venv/bin/python -u run_experiments.py --variety baco_chardonnay_vidal --bands rgb
../../.venv/bin/python -u report.py --variety baco_chardonnay_vidal --bands rgb
../../.venv/bin/python -u plot_architecture.py                                   # figures/architecture.png
# also: --task any both, --no-augmentation (no rotations/flips), --dropout <value>, --backbone <name>, --smoke (quick check)
```
