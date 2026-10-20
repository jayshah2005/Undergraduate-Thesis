# PMMFNet on the vine leaves - everything for this paper

Wang et al., *Progressive multi-scale multi-attention fusion for hyperspectral image classification*,
Sci. Rep. 15:29288 (2025), applied to the 998 leaves (4 classes: healthy / red blotch only / leafroll only / both).

| path | what it is |
|---|---|
| `pmmfnet.py` | the network, read off the paper's figures and text |
| `11_pmmfnet.py` | the original pipeline (unchanged since the original run, so its docstring's `python experiments/...` run line is out of date; see Running below) |
| `runs/original/` | the original 5-fold run: out-of-fold predictions, metrics (.md/.json), logs (`logs/11.md` = full-run log), `checksums.sha256` |
| `controlled.py` | the controlled experiments of `Classification/plan.md` (A1, A2, S1, S2, S3, L2) |
| `run_experiment.py` + `configs/` | **your own runs**: settings in a TOML file (`configs/template.toml` lists every setting; `configs/S1.toml` is an example; `S2_lr3e-4.toml`, `S2_lr3e-6.toml` are your runs), outputs in `runs/custom/<name>/`, `--summary` -> `runs/custom/summary.md` |
| `run_all.sh` | `./run_all.sh smoke` = smoke tests; `./run_all.sh` = all experiments (resumable) + comparison |
| `compare.py` | builds `runs/comparison.md` / `.csv` (one row per experiment, per-fold tables, rerun-noise table, paired comparisons) |
| `runs/<exp>/` | per experiment: `<exp>_oof.csv` (leaf probabilities + fold), `<exp>_metrics.md/.json`, `folds/` (per-fold records and predictions, written as each fold finishes), `checkpoints/fold<k>.pt` (best weights), `fold_assignment.csv`, `log.txt` |
| `runs/smoke/` | logs of the smoke tests run before the controlled experiments |
| `cache/` | fixed evaluation permutations for S2 |
| `REPORT.md` | which configuration is best (S1), how consistent the differences are, what to try next |

Shared code is in `../shared/` because the other pipelines use it too: `common.py` (folds, labels),
`torch_common.py`, `four_class.py` (4-class metrics and report), `logs.py` (the logger every script
reports progress through) and the data cache `../shared/cache/`.

Logging: the training and evaluation functions never print; they take a logger. `run_experiment.py`
writes every message to the terminal and appends it to the run's `log.txt` (plus the traceback if a
run fails). `controlled.py` and `compare.py` log to the terminal only; `run_all.sh` redirects that
into `runs/<exp>/log.txt`, `runs/smoke/<exp>.log` and `runs/compare.log`.
`pmmfnet.py` and `11_pmmfnet.py` are covered by `runs/original/checksums.sha256` and are kept unchanged.

## Running

```bash
cd "claude runs/pmmfnet"
./run_all.sh smoke                                        # ~2 min per experiment
./run_all.sh                                              # all experiments, then compare.py
../../.venv/bin/python -u controlled.py --experiment L2   # one experiment
cp configs/template.toml configs/my_run.toml             # edit it, then:
../../.venv/bin/python -u run_experiment.py configs/my_run.toml --smoke   # ~1 min check
../../.venv/bin/python -u run_experiment.py configs/my_run.toml           # 5 folds, ~40 min
../../.venv/bin/python -u run_experiment.py configs/my_run.toml --set training.lr=3e-5 --set name=lr3e-5
../../.venv/bin/python -u run_experiment.py --summary                     # table of all runs
PYTHONPATH=../shared ../../.venv/bin/python -u 11_pmmfnet.py   # the original pipeline (writes to ../shared/results/)
```

Reproducibility: a fixed seed does not reproduce a run exactly. Same-seed runs with identical settings
differed by up to 0.038 test balanced accuracy in a fold, likely because `cudnn.benchmark=True` (kept
from the original pipeline) picks GPU kernels by timing. `compare.py` treats experiments with identical
per-fold settings as reruns and reports that spread.
