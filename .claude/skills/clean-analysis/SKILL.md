---
name: clean-analysis
description: >-
  Write and refactor the analysis code in this repo (variety_classification/,
  spectral_analysis/, disease_classification/, helpers/); require readable,
  well-documented scripts a newcomer can follow and the shared report format.
  Use when creating or editing any script, README or report in the repo.
---

# Analysis code

## Layout

- Shared code lives in the `helpers/` package: `data.py`, `log.py`, `pca_common.py`, `cnn_common.py`, `diagram.py`, `report_tables.py`, `leaf_sources.csv`. Helpers import each other relatively (`from .data import …`).
- Scripts put the repo root on `sys.path` (`ROOT = HERE.parents[1]`) and import `from helpers.<module> import …`. Code from a sibling folder (e.g. `disease_classification/logistic_regression`) is added to `sys.path` by folder.
- Analysis code reads `vine_hsi.h5` only through `helpers/data.py`, never through `vine_data.py` (kept for `data analysis.ipynb`).
- One topic folder per kind of work: `variety_classification/`, `spectral_analysis/`, `disease_classification/`. New helpers go in `helpers/`, not in a topic folder.
- Leaf means are cached in `cache/` at the repo root.

## Code a newcomer can follow

- Comments and docstrings follow `commenting-python` (`.claude/skills/commenting-python/SKILL.md`).
- How to run a script lives in that folder's README, not a long module docstring.
- One question per folder. Name the folder after that question.
- `main()` is a table of contents: logging → load → compute → write files. No math in `main()`.

## Logs vs logic

- Computation returns values. It never prints and never writes logs.
- Logging uses `helpers/log.py` `get_logger`, set up once in `main()`.
- Keep output short. Analysis and report scripts print one line and keep no log file. Only training keeps a log (latest run, one line per seed or arm), and passing checks stay silent. Never copy a report into a log, or write a file nothing reads.
- Reports and figures are separate functions, called after `compute()`.
- `logs/`, `results/`, `figures/` stay separate.

## Result files

Every run folder (`results/<run>/`, variety and disease, logistic regression and CNN) gets two files, built with `helpers/report_tables.py`, so any two runs can be read side by side:

- `final_report.md`: balanced accuracy only, in the format below.
- `metrics.md`: every metric (balanced accuracy, accuracy, sensitivity (= recall), specificity, precision, F1) with 95% intervals, then confusion counts, then per variety. Same sections as `final_report.md`, tables built with `report_tables.table`.

`final_report.md` format:

- Title `# <Variety|Disease>: <model> [on <varieties>, <bands>]`, then one paragraph: split, score, chance, interval.
- One `##` section per task (disease) or per input (variety), e.g. `## Red blotch (88 + / 162 −)`, `## Full spectrum (204 bands)`.
- Each section opens with `main_table`: `model | bal. acc. [95% CI] | Δ vs logistic regression [95% CI] | Fold 1 (R1) … Fold 5 (R5)`, logistic regression first. Then `fold_note`.
- Disease: one section per task; leafroll and red blotch are always scored separately, never averaged into one number.
- Extra tables follow in this order, only when they apply: per variety (`column_table`), single seeds (`seed_table`), anything else.
- Same names everywhere: `logistic regression`, `CNN, mean pool`, `CNN, attention pool`; another backbone uses its `helpers/backbones.LABEL`, e.g. `small ResNet, mean pool`. Three decimals, intervals as `[lo, hi]`.
- Folds are named `Fold k (Rk)`, never `Rk` alone: each variety has its own rows R1–R5, so a pooled fold tests several field rows.
- One pair per run folder; no separate comparison file.

## Numbers

- PyTorch on the GPU for math (PCA, SNV, sums of squares, permutations). NumPy only at the I/O / plot boundary, with a one-line why.
- Spectral PCA is SNV first unless the user asks for raw.
- Average leaf pixels, then SNV the 204-vector, then PCA. Do not SNV each pixel first.

## Checklist

- [ ] Shared code imported from `helpers/`; nothing imports `vine_data.py`
- [ ] File + function docstrings; non-obvious decisions have a why comment
- [ ] Computation silent; logs only from `main()`
- [ ] Folder named for the question it answers
- [ ] Every run has `final_report.md` and `metrics.md`, built with `report_tables.py` in the shared format
