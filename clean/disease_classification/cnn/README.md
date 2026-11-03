# Whole-leaf CNN with attention pooling

Question: does a small CNN that sees the whole leaf cube find disease better than logistic regression on the
leaf's mean spectrum (`../logistic_regression/`)? By default this runs on Chardonnay (250 leaves), for leafroll and
red blotch. `../REPORT.md` covers those runs.

The leaves, labels, folds, baseline and AUC all come from `../logistic_regression/`, so both models use the same
split and are scored the same way. The baseline takes seconds, so the training log and `report.py` refit it
rather than read a saved copy.

| path | what it is |
|---|---|
| `model.py` | `LeafCNN`: 1×1 spectral conv (204 or 3 bands → 16), four 3×3 conv layers with **masked** batch norm, masked mean or gated-attention pooling, one linear output |
| `train.py` | the two brightness preprocessings, augmentation, training one fold |
| `run_experiments.py` | padding-invariance check → baseline → every (task, arm, seed, fold); skips fold files already on disk |
| `report.py` | metrics (within-row AUC, bootstrap CIs, paired Δ vs baseline) and figures |
| `results/<variety>[_rgb][_no_augmentation]/` | `metrics.md`, `runs/<task>/<arm>/seed<s>/fold<k>.npz` (test probabilities, attention maps, loss curves), `config.json` |
| `figures/<variety>[_rgb][_no_augmentation]/` | `auc_comparison.png`, `attention_maps.png` |
| `logs/<run>.log` | scores of the last training, for a quick look: each task's baseline, then one line per arm (test AUC, per seed, row and variety; best epoch; time). Rerun the same command to rebuild it; finished trainings are skipped |

## Varieties (`--variety`)

One of `Baco`, `"Cab Franc"`, `Chardonnay` (default) or `Vidal`, or `all` to train one model on all 998 leaves.
Results go to `results/<variety>/`, e.g. `results/chardonnay/` or `results/all/`, the same names as the baseline's.
Give `report.py` the same `--variety`.

With `all`, AUCs compare leaves of the same variety as well, so recognising a variety can't raise them, and
`metrics.md` adds AUC per variety. The model can still learn variety as a stand-in for disease while training
(Cab Franc is 96% red blotch), and balanced accuracy is pooled, so use the AUCs.

## Bands (`--bands`)

`all` (default) feeds the network all 204 bands. `rgb` feeds it only the bands nearest 640, 550 and 460 nm, as in
the variety CNN, into `results/<variety>_rgb/`. The leaf mask still comes from all 204 bands, so a leaf covers the
same pixels either way. The baseline in the log and in `report.py` uses the same bands as the CNN. Under `rgb`, pixel
SNV works over 3 values. Give `report.py` the same `--bands`.

## Tasks (`--task`)

Give one or more. The default is `leafroll red_blotch`, each trained on its own. The tasks and the rule that leaves
out a variety with too few positive or negative leaves are in `../logistic_regression/README.md`. If nothing is
left, or a fold's training, validation or test set would hold only one class, the run stops before training.

`report.py` covers every task with runs in the folder, so a new task never removes an earlier one.

## Set-up in one paragraph

Each leaf cube (204 bands × its own height × width) is centred on one square canvas (the largest leaf side
among the leaves loaded, 136 × 136 for Chardonnay, square so 90° rotations fit); the padding is 0 and a mask marks leaf pixels.
Normalisation statistics and pooling use leaf positions only, so padding cannot change a prediction
(checked at start-up: shifting a leaf or enlarging the canvas changes its logit by < 1e-6).
Arms = brightness handling {per-band z-score, per-pixel SNV} × pooling {mean, attention}.
Training augmentation is random 90° rotations and flips (square canvas, so they fit), with no brightness or band
jitter. `--no-augmentation` turns them off, into `results/<variety>_no_augmentation/`. 3 seeds per arm. Test = row k, validation (early stopping) = row k+1, train = the other 3 rows.

## Running

```bash
cd clean/disease_classification/cnn
../../../.venv/bin/python -u run_experiments.py --smoke   # ~15 s: checks + 1 fold of every arm
../../../.venv/bin/python -u run_experiments.py           # 120 trainings, ~6 min (resumable)
../../../.venv/bin/python -u report.py                    # results/chardonnay/metrics.md + figures/chardonnay/

# another dropout (default 0.5): give both scripts --dropout; the folder name gets _dropout0.2
../../../.venv/bin/python -u run_experiments.py --dropout 0.2

# without rotations and flips: give both scripts the flag; results/chardonnay_no_augmentation/
../../../.venv/bin/python -u run_experiments.py --no-augmentation
../../../.venv/bin/python -u report.py --no-augmentation

# RGB bands only: give both scripts --bands rgb; results/chardonnay_rgb/
../../../.venv/bin/python -u run_experiments.py --bands rgb
../../../.venv/bin/python -u report.py --bands rgb

# another variety or task: results/baco/
../../../.venv/bin/python -u run_experiments.py --variety Baco --task both any
../../../.venv/bin/python -u report.py --variety Baco

# all varieties pooled: results/all/
../../../.venv/bin/python -u run_experiments.py --variety all --task any
../../../.venv/bin/python -u report.py --variety all
```

Older runs (with brightness jitter, dropout 0.2, other fold designs) were deleted; `../REPORT.md` keeps their numbers.

A results folder only resumes runs made with the same training settings (`config.json`); otherwise
`run_experiments.py` stops and asks for the folder to be moved or deleted.
