---
name: separate-logging
description: Use whenever writing or refactoring any script in this repo. Keeps logging/printing/report writing out of the computation code, and requires comments that let a newcomer follow the logic.
---

# Separate logs from logic, and comment for a newcomer

## 1. Logs are not logic

- **Computation functions never print and never write log files.** They take inputs and *return*
  results (numbers, arrays, dicts, dataclasses). A function that trains, scores or transforms data
  must be callable silently from a notebook or a test.
- **Progress messages go through `logging`**, set up once in the script's entry point with the
  shared helper `claude runs/shared/logs.py`:

  ```python
  from logs import get_logger
  log = get_logger("pmmfnet", log_file=out / "log.txt")   # terminal + file
  log.info("fold %d: test bal. acc. %.3f", fold, bacc)
  ```

  Long-running loops that must report progress (e.g. one line per epoch) take a `log` argument
  or a callback instead of calling `print` - the caller decides where it goes.
- **Reports and tables are their own functions** (e.g. `to_markdown(results)`, `write_outputs(...)`),
  ideally in their own section or module (`report.py`), called from `main()` after the computation.
- **Log files live in a `logs/` folder (or the run's own folder)**, never mixed into `results/`.
- `main()` reads like a table of contents: parse arguments -> set up logging -> load data ->
  compute -> write reports. No computation inside `main()` beyond calling functions.
- Keep the printed/written text the same when refactoring (downstream tools grep for lines such as
  `[smoke]`), unless the user asked to change it.

## 2. Comments a newcomer can follow

- **Every file starts with a docstring**: what question it answers, how to run it, what it reads,
  what it writes.
- **Every function has a docstring**: what it does, its inputs/outputs with shapes and units where
  relevant (`(n_leaves, 204) float64 reflectance`).
- **Comment the *why*, not the *what***: why rows are held out together, why labels are shuffled
  within rows, why SNV, why the scaler is fitted on training leaves only. Skip comments that only
  repeat the code.
- **Name the steps of long functions** with short section comments (`# 1. inner CV picks C`).
- **Define domain terms once** (SNV, within-variety AUC, fold = row, RB / LR) in the file docstring.
- Keep comment density consistent with the surrounding code; no essays.

## Checklist before finishing

- [ ] `grep -n "print(" <file>` only finds lines in the entry point / reporting code (or none).
- [ ] Logic functions return values; no logging side effects inside them.
- [ ] Log files go to `logs/` or the run folder.
- [ ] File + function docstrings present; the non-obvious decisions have a *why* comment.
- [ ] Behaviour and outputs unchanged (rerun and compare when refactoring).
