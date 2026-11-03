# Whole-leaf CNN

Does a small CNN that sees the whole leaf find disease better than the mean-spectrum baseline
(`../logistic_regression/`)? Leaves, labels, folds, baseline and scoring all come from that folder.

| file | what it is |
|---|---|
| `model.py` | `LeafCNN`: 1×1 band mix → four 3×3 conv layers → mean or attention pooling over leaf pixels → one output |
| `train.py` | input normalisation, rotations and flips, training one fold (early stopping on the validation row) |
| `run_experiments.py` | trains every task × model × seed × fold; skips folds already saved; writes `logs/<run>.log` |
| `report.py` | `metrics.md` and figures for one run |
| `results/<run>/` | `metrics.md`, `config.json`, saved test predictions per fold |
| `figures/<run>/` | `balanced_accuracy.png`, `attention_maps.png` |

**Models.** Four per task: input normalised per band (`band_z`) or per pixel (`snv`), × mean or attention pooling.
Each is trained with 3 seeds; the reported score uses their average probability.

**Score.** Balanced accuracy at P ≥ 0.5, as for the baseline, with a 95% interval and the paired difference (Δ) to
the baseline.

**Run names.** `<variety>[_rgb][_no_augmentation]`, e.g. `chardonnay`, `chardonnay_rgb`, `all`. Give `report.py`
the same flags as `run_experiments.py`.

```bash
cd clean/disease_classification/cnn
../../../.venv/bin/python -u run_experiments.py                              # Chardonnay, leafroll + red blotch, ~6 min
../../../.venv/bin/python -u report.py
../../../.venv/bin/python -u run_experiments.py --bands rgb                  # RGB bands only
../../../.venv/bin/python -u report.py --bands rgb
../../../.venv/bin/python -u run_experiments.py --variety all --task any     # all 998 leaves, ~20 min
../../../.venv/bin/python -u report.py --variety all
# also: --no-augmentation (no rotations/flips), --dropout <value>, --smoke (quick check)
```
