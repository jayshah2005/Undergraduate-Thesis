# Whole-leaf CNN, one variety at a time

Does a CNN trained on one variety's leaves only find disease better than that variety's mean-spectrum baseline? One
model per variety: Baco, Chardonnay and Vidal. Cab Franc sits out (1 leaf with leafroll, 9 without red blotch).

The network, training and scoring are `../cnn/`'s (same 212 × 212 canvas, SNV per pixel, mean or attention pooling,
3 seeds, balanced accuracy at P ≥ 0.5); these scripts only loop over the varieties.

| file | what it is |
|---|---|
| `run_within_variety.py` | trains each variety's task × pooling × seed × fold; skips folds already saved; writes `logs/<run>.log` |
| `report_within_variety.py` | `final_report.md`, `metrics.md` and figures for each variety |
| `plot_architecture.py` | poster schematic of the network → `figures/architecture.png` |
| `results/<variety>[_rgb]/` | `final_report.md` (balanced accuracy), `metrics.md` (every metric), `config.json`, saved test predictions per fold |
| `figures/<variety>[_rgb]/` | `balanced_accuracy.png`, `attention_maps.png` |

```bash
cd disease_classification/within_variety_disease_cnn
../../.venv/bin/python -u run_within_variety.py                  # Baco, Chardonnay, Vidal; leafroll + red blotch
../../.venv/bin/python -u report_within_variety.py
../../.venv/bin/python -u run_within_variety.py --bands rgb      # RGB bands only
../../.venv/bin/python -u report_within_variety.py --bands rgb
../../.venv/bin/python -u plot_architecture.py
# also: --variety Vidal (one or more), --task any both, --no-augmentation, --dropout <value>, --smoke
```
