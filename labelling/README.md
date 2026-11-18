# Labelling

How `vine_hsi.h5` got its leaves and virus labels, and a tool to check the labels by hand. Run everything from this
folder with the repo's `.venv`.

| file | what it does |
|---|---|
| `review_labels.py` | puts the lab sheets and the HDF5 side by side, runs the label checks, lists every changed leaf with its test history, then stops at a breakpoint so you can explore |
| `update_labels_final_2026.py` | writes the FINAL 2026 labels (sheet columns E and M) into an HDF5; dry run unless `--apply` |
| `build_h5_single.py` | builds `vine_hsi.h5` from `Data/` with the interim labels (then run the update above) |
| `build_h5_mirror.py` | the same, as one HDF5 file per leaf |
| `restore_leaf_headers.py` | one-off (2026-10-29): rebuilt the leaf `.hdr` files from the camera captures and wrote `../helpers/leaf_sources.csv` |
| `docs/SOURCE.md` | camera and wavelength notes |

Not in git (see `.gitignore`): the lab sheets (`New Testing Results.xlsx`, the FINAL 2026 labels;
`Virus testing data for interim report.xlsx`, the interim labels), because they name testers and hold unpublished
results; the raw leaf cubes in `Data/`; and the HDF5 copies (`vine_hsi.h5`, an identical copy of `../vine_hsi.h5`,
and `vine_hsi_interim_labels.h5`, the file before the relabel).

## Review the labels

```bash
../.venv/bin/python review_labels.py             # checks + summary, then a breakpoint
../.venv/bin/python review_labels.py --no-break  # checks + summary only
```

At the breakpoint (`(Pdb)` prompt):

| type | to see |
|---|---|
| `df[df.changed]` | the 7 relabelled leaves (`df`: one row per vine in the sheet, every test column, its reason colour, the HDF5 labels) |
| `show("Vidal/R3/V5")` | every test and label for one leaf, under the sheet's own column names |
| `look("Vidal/R3/V5")` | the leaf's image and mean spectrum next to its variety's mean |
| `interact` | a normal Python prompt with all of the above (Ctrl-D goes back) |
| `c` | finish |

In VS Code, run the script under the debugger instead to see `df` in the variables pane.

## Which labels the models use

The FINAL 2026 columns, the lab's decision after retesting (Baco and Vidal rows R1–R3). The colour of a FINAL cell
says why it differs from the first 2026 test (legend in row 1 of the sheet). Compared with the interim labels, 7 Vidal
leaves differ; each vine keeps its old values as `red_blotch_interim` / `leafroll3_interim`, and
`vine_hsi_interim_labels.h5` is the whole file as it was before the relabel.
