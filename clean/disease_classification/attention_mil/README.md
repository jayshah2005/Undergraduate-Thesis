# Whole-leaf CNN with attention pooling (Chardonnay)

Question: inside one variety (Chardonnay, 250 leaves), does a small CNN that sees the whole leaf cube
find leafroll or red blotch better than a logistic regression on the leaf's mean spectrum?

| path | what it is |
|---|---|
| `model.py` | `LeafCNN`: 1×1 spectral conv (204 → 16), four 3×3 conv layers with **masked** batch norm, masked mean or gated-attention pooling, one linear output |
| `train.py` | padding onto the canvas, the two brightness preprocessings, augmentation, the fold schemes (`FOLD_SCHEMES`), training one fold, the logistic baseline |
| `run_experiments.py` | padding-invariance check → baseline → every (task, arm, seed, fold) for one fold scheme; skips fold files already on disk |
| `report.py` | metrics (within-fold AUC, bootstrap CIs, paired Δ vs baseline) for one scheme, or `--compare` across schemes |
| `REPORT.md` | what came out and what it means |
| `results/<scheme>/` | `baseline.npz`, `runs/<task>/<arm>/seed<s>/fold<k>.npz` (test probabilities, attention maps, loss curves), `metrics.md`, `metrics.json` (read by `--compare`), `config.json` |
| `results/fold_comparison.md` | every model under every fold scheme, side by side |
| `figures/` | `row*/auc_comparison.png`, `row*/attention_maps.png`, `fold_comparison*.png` |
| `logs/<run>.log` | progress of the last training: one line per task × arm |

## Fold schemes (`--folds`)

Every scheme keeps a photo (the camera image the leaves were cut from, from `../../leaf_sources.csv`) inside one
fold. Rows R2–R5 have 10 photos of 5 leaves; R1 has 12 of 4 and one of 2. Test = fold k,
validation (early stopping) = fold k+1, train = the other 3. AUCs compare leaves of the same fold
**and the same row** only, so a row's disease rate cannot inflate them.

| `--folds` | fold k holds | question it answers |
|---|---|---|
| `row` (default) | field row k | does it work on a part of the vineyard it has never seen? |
| `photo` | random photos dealt evenly per row: 2 per fold (R1: 3 or 2) | does it work on new vines in rows it has seen? |
| `block` | the photos of each row whose first vine is in 10k+1 … 10k+10 | same as `photo`, but held-out vines are not next to training vines (except at block edges), so vine-to-vine spread of the virus and drift in imaging order help less |

## Set-up in one paragraph

Each leaf cube (204 bands × its own height × width) is centred on a 136 × 136 canvas (the largest
Chardonnay leaf side, square so 90° rotations fit); the padding is 0 and a mask marks leaf pixels.
Normalisation statistics and pooling use leaf positions only, so padding cannot change a prediction
(checked at start-up: shifting a leaf or enlarging the canvas changes its logit by < 1e-6).
Arms = brightness handling {per-band z-score + brightness jitter, per-pixel SNV} × pooling {mean, attention}.
The brightness jitter (leaf × U(0.8, 1.2) during training) can be turned off with `--no-brightness-jitter`.
Tasks are binary: leafroll present (LR only + both) and red blotch present (RB only + both).
Folds: one of the three schemes above. 3 seeds per arm. Baseline = StandardScaler + class-balanced
logistic regression on the SNV leaf-mean spectrum, same folds, trained on 4 folds (and on the CNN's
3 training folds for reference).

## Running

```bash
cd clean/disease_classification/attention_mil
../../../.venv/bin/python -u run_experiments.py --folds row --smoke   # ~15 s: checks + 1 fold of every arm
../../../.venv/bin/python -u run_experiments.py --folds row           # 120 trainings, ~6 min (resumable)
../../../.venv/bin/python -u report.py --folds row                    # results/row/metrics.md + figures/row/
# same for --folds photo and --folds block, then:
../../../.venv/bin/python -u report.py --compare                      # results/fold_comparison.md + figure

# without brightness jitter: give both scripts the flag; output goes to results/row_no_brightness_jitter/
../../../.venv/bin/python -u run_experiments.py --folds row --no-brightness-jitter
../../../.venv/bin/python -u report.py --folds row --no-brightness-jitter
../../../.venv/bin/python -u report.py --compare --no-brightness-jitter   # results/fold_comparison_no_brightness_jitter.md

# another dropout (default 0.5): give all three commands --dropout; the folder name gets _dropout0.2
../../../.venv/bin/python -u run_experiments.py --folds row --no-brightness-jitter --dropout 0.2
```

Current results use `--no-brightness-jitter`. Older runs (with jitter, dropout 0.2) were deleted; `REPORT.md` keeps their numbers.

A results folder only resumes runs made with the same training settings (`config.json`); otherwise
`run_experiments.py` stops and asks for the folder to be moved or deleted.
