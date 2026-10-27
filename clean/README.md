# clean/

Each vine is one leaf mean spectrum from `vine_hsi.h5`. `data.py` caches the means in `cache/`.

| path | what it is |
|---|---|
| `data.py` | one mean spectrum per vine from the HDF5 |
| `log.py` | terminal + `logs/run.log` |
| `PCA/` | two PCA questions (variety; disease class) |
| `../.claude/skills/clean-analysis/` | how code in this folder must be written |

Run scripts from inside their folder with `../../../.venv/bin/python -u <script>` (from `clean/PCA/<question>/`).
