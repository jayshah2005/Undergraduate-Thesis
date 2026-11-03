# Whole-leaf CNN with attention pooling

Question: does a small CNN that sees the whole leaf cube find disease better than a logistic regression on
the leaf's mean spectrum? By default this runs on Chardonnay (250 leaves), for leafroll and red blotch.
`REPORT.md` covers those runs.

| path | what it is |
|---|---|
| `model.py` | `LeafCNN`: 1×1 spectral conv (204 → 16), four 3×3 conv layers with **masked** batch norm, masked mean or gated-attention pooling, one linear output |
| `train.py` | padding onto the canvas, the two brightness preprocessings, augmentation, the row split, training one fold, the logistic baseline |
| `run_experiments.py` | padding-invariance check → baseline → every (task, arm, seed, fold); skips fold files already on disk |
| `scoring.py` | the within-group AUC, used by both the training log and the report |
| `report.py` | metrics (within-row AUC, bootstrap CIs, paired Δ vs baseline) and figures |
| `REPORT.md` | what came out and what it means |
| `results/[<variety>_]row*/` | `metrics.md`, `baseline.npz`, `runs/<task>/<arm>/seed<s>/fold<k>.npz` (test probabilities, attention maps, loss curves), `config.json` |
| `figures/[<variety>_]row*/` | `auc_comparison.png`, `attention_maps.png` |
| `logs/<run>.log` | scores of the last training, for a quick look: each task's baseline, then one line per arm (test AUC, per seed, row and variety; best epoch; time). Rerun the same command to rebuild it; finished trainings are skipped |

## Varieties (`--variety`)

One of `Baco`, `"Cab Franc"`, `Chardonnay` (default) or `Vidal`, or `all` to train one model on all 998 leaves.
Anything other than Chardonnay starts the folder name, e.g. `results/all_row/`. Give `report.py` the same `--variety`.

With `all`, AUCs compare leaves of the same variety as well, so recognising a variety can't raise them, and
`metrics.md` adds AUC per variety. The model can still learn variety as a stand-in for disease while training
(Cab Franc is 96% red blotch), and balanced accuracy is pooled, so use the AUCs.

## Tasks (`--task`)

Give one or more. The default is `leafroll red_blotch`, each trained on its own.

| `--task` | positive | negative |
|---|---|---|
| `leafroll` | leafroll present | no leafroll |
| `red_blotch` | red blotch present | no red blotch |
| `both` | both diseases present | everything else, including leaves with only one |
| `any` | any disease present | healthy |

A task leaves out any variety with fewer than 5 positive or 5 negative leaves. Under `all`, that means Cab Franc
sits out `leafroll` and `both` (1 positive leaf each), and the log says so. If nothing is left, or a fold's training,
validation or test set would hold only one class, the run stops before training. Cab Franc on `red_blotch` or `any`
is allowed, but it has only 9 healthy leaves, so expect wide intervals.

`report.py` covers every task with runs in the folder, so a new task never removes an earlier one.

## Folds

Fold k is field row k. Test = row k, validation (early stopping) = row k+1, train = the other 3 rows. AUCs compare
leaves of the same row only, so a row's disease rate cannot inflate them. Every photo (the camera image the leaves
were cut from) lies inside one row, so its shared lighting never spans training and test.

## Set-up in one paragraph

Each leaf cube (204 bands × its own height × width) is centred on one square canvas (the largest leaf side
among the leaves loaded, 136 × 136 for Chardonnay, square so 90° rotations fit); the padding is 0 and a mask marks leaf pixels.
Normalisation statistics and pooling use leaf positions only, so padding cannot change a prediction
(checked at start-up: shifting a leaf or enlarging the canvas changes its logit by < 1e-6).
Arms = brightness handling {per-band z-score + brightness jitter, per-pixel SNV} × pooling {mean, attention}.
The brightness jitter (leaf × U(0.8, 1.2) during training) can be turned off with `--no-brightness-jitter`.
Tasks are binary (see above).
3 seeds per arm. Baseline = StandardScaler + class-balanced logistic regression on the SNV leaf-mean spectrum,
trained on the CNN's 3 training rows, so neither model has more data. It has no setting to choose, so it doesn't
use the validation row.

## Running

```bash
cd clean/disease_classification/attention_mil
../../../.venv/bin/python -u run_experiments.py --smoke   # ~15 s: checks + 1 fold of every arm
../../../.venv/bin/python -u run_experiments.py           # 120 trainings, ~6 min (resumable)
../../../.venv/bin/python -u report.py                    # results/row/metrics.md + figures/row/

# without brightness jitter: give both scripts the flag; output goes to results/row_no_brightness_jitter/
../../../.venv/bin/python -u run_experiments.py --no-brightness-jitter
../../../.venv/bin/python -u report.py --no-brightness-jitter

# another dropout (default 0.5): give both scripts --dropout; the folder name gets _dropout0.2
../../../.venv/bin/python -u run_experiments.py --no-brightness-jitter --dropout 0.2

# another variety or task: results/baco_row_no_brightness_jitter/
../../../.venv/bin/python -u run_experiments.py --variety Baco --task both any --no-brightness-jitter
../../../.venv/bin/python -u report.py --variety Baco --no-brightness-jitter

# all varieties pooled: results/all_row_no_brightness_jitter/; about 8 min per task × arm, so ~30 min per task
../../../.venv/bin/python -u run_experiments.py --variety all --task any --no-brightness-jitter
../../../.venv/bin/python -u report.py --variety all --no-brightness-jitter
```

Current results use `--no-brightness-jitter`. Older runs (with jitter, dropout 0.2, other fold designs) were deleted; `REPORT.md` keeps their numbers.

A results folder only resumes runs made with the same training settings (`config.json`); otherwise
`run_experiments.py` stops and asks for the folder to be moved or deleted.
