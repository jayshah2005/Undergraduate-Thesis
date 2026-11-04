# Whole-leaf CNN, pooled varieties

Does one small CNN trained on the leaves of several varieties find disease better than the mean-spectrum baseline
(`../logistic_regression/`)? Leaves, labels, folds, baseline and scoring all come from that folder. The network and
training code here are also used by `../within_variety_disease_cnn/`.

| file | what it is |
|---|---|
| `model.py` | `LeafCNN`: 1×1 band mix → four 3×3 conv layers → mean or attention pooling over leaf pixels → one output |
| `train.py` | SNV per pixel, rotations and flips, training one fold (early stopping on the validation row); `CANVAS = 212` |
| `run_experiments.py` | trains every task × pooling × seed × fold of one pool; skips folds already saved; writes `logs/<run>.log` |
| `report.py` | `metrics.md` and figures for one run |
| `plot_architecture.py` | poster schematic of the network → `figures/architecture.png` |
| `results/<run>/` | `metrics.md`, `config.json`, saved test predictions per fold |
| `figures/<run>/` | `balanced_accuracy.png`, `attention_maps.png` |

**Pools.** `all` (998 leaves) or `baco_chardonnay_vidal` (748). A variety with fewer than 5 positive or negative
leaves sits out a task, so Cab Franc drops out of leafroll and both pools train leafroll on the same 748 leaves.

**Models.** Each pixel's spectrum is SNV-normalised, as in the variety CNN, on a 212 × 212 canvas. Two models per task:
mean pooling or attention pooling over leaf pixels. Each is trained with 3 seeds; the score uses their average probability.

**Score.** Balanced accuracy at P ≥ 0.5, as for the baseline, with a 95% interval, the paired difference (Δ) to the
baseline, and the score on each variety's leaves.

**Run names.** `<pool>[_rgb][_no_augmentation]`, e.g. `all`, `baco_chardonnay_vidal_rgb`. Give `report.py` the same
flags as `run_experiments.py`.

```bash
cd clean/disease_classification/cnn
../../../.venv/bin/python -u run_experiments.py                                     # all 998 leaves, leafroll + red blotch
../../../.venv/bin/python -u report.py
../../../.venv/bin/python -u run_experiments.py --variety baco_chardonnay_vidal --bands rgb
../../../.venv/bin/python -u report.py --variety baco_chardonnay_vidal --bands rgb
../../../.venv/bin/python -u plot_architecture.py                                   # figures/architecture.png
# also: --task any both, --no-augmentation (no rotations/flips), --dropout <value>, --smoke (quick check)
```
