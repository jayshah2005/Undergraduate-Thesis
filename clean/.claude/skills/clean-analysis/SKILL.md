---
name: clean-analysis
description: >-
  Write and refactor code under clean/. Keep clean/ independent of claude runs/
  and other experiment folders; require readable, well-documented scripts a
  newcomer can follow. Use when creating or editing files in clean/, clean/PCA,
  or when the user mentions the clean folder.
---

# clean/ analysis

Analyses in `clean/` have their own loaders. They do not import `claude runs/`,
`vine_data.py`, or other experiment folders.

## Isolation

- Import only from `clean/` (`data.py`, `log.py`, `PCA/pca_common.py`) and third-party libraries.
- Never `sys.path` into `claude runs/` or `from common import …` / `from spectra import …` / `from vine_data import …`.
- The only file outside `clean/` that code may read is `vine_hsi.h5`, through `clean/data.py`.
- Cache leaf means in `clean/cache/`, not in `claude runs/shared/cache/`.
- Do not add experiment code under `claude runs/` while working on `clean/`.

## Code a newcomer can follow

- Comments and docstrings follow `commenting-python` (`clean/.claude/skills/commenting-python/SKILL.md`).
- How to run a script lives in that folder's README, not a long module docstring.
- One question per folder. Name the folder after that question.
- `main()` is a table of contents: logging → load → compute → write files. No math in `main()`.

## Logs vs logic

- Computation returns values. It never prints and never writes logs.
- Logging uses `clean/log.py` `get_logger`, set up once in `main()`.
- Reports and figures are separate functions, called after `compute()`.
- `logs/`, `results/`, `figures/` stay separate.

## Numbers

- PyTorch on the GPU for math (PCA, SNV, sums of squares, permutations). NumPy only at the I/O / plot boundary, with a one-line why.
- Spectral PCA in `clean/` is SNV first unless the user asks for raw.
- Average leaf pixels, then SNV the 204-vector, then PCA. Do not SNV each pixel first.

## Checklist

- [ ] No imports from `claude runs/` or `vine_data.py`
- [ ] File + function docstrings; non-obvious decisions have a why comment
- [ ] Computation silent; logs only from `main()`
- [ ] Folder named for the question it answers
