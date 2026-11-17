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
| `plot_architecture.py` | poster schematics of the network → `figures/architecture.png` (small CNN) and `figures/architecture_<backbone>.png` for each other backbone, drawn at the same scale |
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

The layers before the pooling come from `helpers/backbones.py` and are picked with `--backbone`. Every backbone uses masked batch norm and puts leaves on a grid of its own total stride (4×; 32× for `resnet18`, on a 224 canvas), so the padding check still passes; the check runs in float64 so it tests the masking, not rounding. All train with the same settings. The default backbone's report also shows every other backbone that was run, `SECOND_BACKBONE` from `model.py` first. `small_resnet` won a validation-row screen against three other published backbones (HybridSN 3D CNN, 1D spectral CNN, ImageNet ResNet-18), which were then removed; they are in git history at `fb310b9`.

| `--backbone` | from | changed for whole leaves |
|---|---|---|
| `small_cnn` (default) | this repo's own design | — |
| `small_resnet` | residual blocks, He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR | 4 basic blocks (32, 64, 64, 64 channels) instead of ResNet-18's 8 |
| `small_resnet_strided` | `small_resnet` without max pooling: stride-2 first convs, as He et al. downsample; Springenberg et al. 2015, "Striving for Simplicity: The All Convolutional Net", ICLR workshop | the two max pools become stride-2 convs, with a 1×1 stride-2 conv + BN on those blocks' skips |
| `resnet18` | the full 18-layer network of He et al. 2016 (Table 1), from scratch | the fewest changes: the first conv reads the bands, masked BN, leaf pooling; it shrinks the leaf 32×, so leaves sit on a 224 canvas at offsets that are multiples of 32, and a position stays leaf if any of it is leaf (all-leaf would leave 212 of 998 leaves with none) |

```bash
cd disease_classification/cnn
../../.venv/bin/python -u run_experiments.py                                     # all 998 leaves, leafroll + red blotch
../../.venv/bin/python -u report.py
../../.venv/bin/python -u run_experiments.py --variety baco_chardonnay_vidal --bands rgb
../../.venv/bin/python -u report.py --variety baco_chardonnay_vidal --bands rgb
../../.venv/bin/python -u plot_architecture.py                                   # figures/architecture*.png
# also: --task any both, --no-augmentation (no rotations/flips), --dropout <value>, --backbone <name>, --smoke (quick check)
```
